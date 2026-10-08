// Experimental Float64 local Stokes assembly + cuBLAS batched pivoted LU.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <string>
// RBFLAB_GENERATED_SPACE
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void blas(cublasStatus_t e){if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS status "+std::to_string(e));}
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
struct Node{int op;double x,y;};
struct Task{double mu,x,y;std::vector<double> v,p;std::vector<Node> nodes;};
template<class T>struct Device{
 T* p=nullptr;size_t count;
 explicit Device(size_t n):count(n){check(cudaMalloc((void**)&p,sizeof(T)*n));}
 ~Device(){if(p)cudaFree(p);}
 Device(const Device&)=delete;Device& operator=(const Device&)=delete;
 void upload(const T* src){check(cudaMemcpy(p,src,count*sizeof(T),cudaMemcpyHostToDevice));}
 void download(T* dst){check(cudaMemcpy(dst,p,count*sizeof(T),cudaMemcpyDeviceToHost));}
};
__global__ void assemble(int n,int count,int nv,int np,const Node* nodes,const double* params,const double* meta,double* G,double* B){
 int total=count*n*(n+4);
 for(int index=blockIdx.x*blockDim.x+threadIdx.x;index<total;index+=blockDim.x*gridDim.x){
  int t=index/(n*(n+4)),entry=index%(n*(n+4));
  KernelSpec v{params+t*(nv+np)},p{params+t*(nv+np)+nv};const Node* pts=nodes+t*n;
  if(entry<n*n){int row=entry%n,col=entry/n;if(row<col)continue;
   double value=custom_eval(pts[col].op,pts[row].op,pts[row].x-pts[col].x,pts[row].y-pts[col].y,meta[3*t],v,p);
   G[t*n*n+col*n+row]=value;G[t*n*n+row*n+col]=value;
  }else{int j=entry-n*n,row=j%n,target=j/n;
   B[t*n*4+target*n+row]=custom_eval(pts[row].op,target+5,meta[3*t+1]-pts[row].x,meta[3*t+2]-pts[row].y,meta[3*t],v,p);
  }
 }
}
__global__ void scales(int n,int count,const double* G,double* scale){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count*n;i+=gridDim.x*blockDim.x){int t=i/n,row=i%n;double mx=0;for(int j=0;j<n;j++)mx=fmax(mx,fabs(G[t*n*n+j*n+row]));scale[i]=mx>0?1/sqrt(mx):nan("");}
}
__global__ void equilibrate(int n,int count,const double* G,const double* B,const double* scale,double* A,double* R){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count*n*(n+4);i+=gridDim.x*blockDim.x){int t=i/(n*(n+4)),j=i%(n*(n+4));if(j<n*n)A[t*n*n+j]=G[t*n*n+j]*scale[t*n+j%n]*scale[t*n+j/n];else{j-=n*n;R[t*n*4+j]=B[t*n*4+j]*scale[t*n+j%n];}}
}
__global__ void unscale(int n,int count,const double* scale,double* W){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count*n*4;i+=gridDim.x*blockDim.x){int t=i/(n*4);W[i]*=scale[t*n+i%n];}}
__global__ void residuals(int n,int count,const double* G,const double* B,const double* W,double* error,double* denom){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count*n*4;i+=gridDim.x*blockDim.x){int t=i/(n*4),j=i%(n*4),row=j%n,k=j/n;double d=-B[i];for(int col=0;col<n;col++)d+=G[t*n*n+col*n+row]*W[t*n*4+k*n+col];double a=isfinite(d)?fabs(d):INFINITY;atomicMax((unsigned long long*)(error+t),(unsigned long long)__double_as_longlong(a));atomicMax((unsigned long long*)(denom+t),(unsigned long long)__double_as_longlong(fabs(B[i])));}
}
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("Usage: cuda_lhi INPUT OUTPUT");
 std::ifstream in(argv[1]);int count,chunk,nv,np;if(!(in>>count>>chunk>>nv>>np)||count<1||chunk<1||nv<0||np<0)throw std::runtime_error("Invalid header");
 std::vector<Task> tasks(count);std::map<int,std::vector<int>> groups;
 for(int i=0;i<count;i++){auto&t=tasks[i];int n;in>>n>>t.mu>>t.x>>t.y;if(n<1||n>4096)throw std::runtime_error("Invalid matrix size");t.v.resize(nv);t.p.resize(np);for(auto&v:t.v)in>>v;for(auto&v:t.p)in>>v;t.nodes.resize(n);for(auto&v:t.nodes)in>>v.op>>v.x>>v.y;if(!in)throw std::runtime_error("Invalid task input");groups[n].push_back(i);}
 auto initial=Clock::now();check(cudaFree(0));cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,0));cublasHandle_t handle;blas(cublasCreate(&handle));double init=elapsed(initial);
 std::vector<std::vector<double>> weights(count);std::vector<double> residual(count);double h2d=0,assembly=0,solve=0,d2h=0;int batches=0;
 auto begin=Clock::now();
 for(auto&group:groups){int n=group.first;auto&ids=group.second;
  for(size_t offset=0;offset<ids.size();offset+=chunk){int b=std::min(size_t(chunk),ids.size()-offset);batches++;
   std::vector<Node> nodes;std::vector<double> params,meta;
   for(int j=0;j<b;j++){auto&t=tasks[ids[offset+j]];nodes.insert(nodes.end(),t.nodes.begin(),t.nodes.end());params.insert(params.end(),t.v.begin(),t.v.end());params.insert(params.end(),t.p.begin(),t.p.end());meta.insert(meta.end(),{t.mu,t.x,t.y});}
   Device<Node> dn(b*n);Device<double> dp(std::max(1,b*(nv+np))),dm(b*3),G(size_t(b)*n*n),A(size_t(b)*n*n),B(b*n*4),W(b*n*4),scale(b*n),error(b),denom(b);Device<int> piv(b*n),info(b);Device<double*> aa(b),ww(b);
   std::vector<double*> ap(b),wp(b);for(int j=0;j<b;j++){ap[j]=A.p+size_t(j)*n*n;wp[j]=W.p+j*n*4;}
   auto t=Clock::now();dn.upload(nodes.data());if(!params.empty())check(cudaMemcpy(dp.p,params.data(),params.size()*sizeof(double),cudaMemcpyHostToDevice));dm.upload(meta.data());aa.upload(ap.data());ww.upload(wp.data());check(cudaMemset(error.p,0,b*sizeof(double)));check(cudaMemset(denom.p,0,b*sizeof(double)));h2d+=elapsed(t);
   int blocks=std::min(4096,(b*n*(n+4)+127)/128);t=Clock::now();
   assemble<<<blocks,128>>>(n,b,nv,np,dn.p,dp.p,dm.p,G.p,B.p);scales<<<blocks,128>>>(n,b,G.p,scale.p);equilibrate<<<blocks,128>>>(n,b,G.p,B.p,scale.p,A.p,W.p);check(cudaGetLastError());check(cudaDeviceSynchronize());assembly+=elapsed(t);
   t=Clock::now();blas(cublasDgetrfBatched(handle,n,aa.p,n,piv.p,info.p,b));std::vector<int> status(b);info.download(status.data());for(int j=0;j<b;j++)if(status[j])throw std::runtime_error("LU failure at stencil "+std::to_string(ids[offset+j])+", info="+std::to_string(status[j]));int result=0;blas(cublasDgetrsBatched(handle,CUBLAS_OP_N,n,4,(const double**)aa.p,n,piv.p,ww.p,n,&result,b));if(result)throw std::runtime_error("Batched solve failed");unscale<<<blocks,128>>>(n,b,scale.p,W.p);residuals<<<blocks,128>>>(n,b,G.p,B.p,W.p,error.p,denom.p);check(cudaGetLastError());check(cudaDeviceSynchronize());solve+=elapsed(t);
   t=Clock::now();std::vector<double> host(b*n*4),err(b),norm(b);W.download(host.data());error.download(err.data());denom.download(norm.data());d2h+=elapsed(t);
   for(int j=0;j<b;j++){int id=ids[offset+j];weights[id].assign(host.begin()+j*n*4,host.begin()+(j+1)*n*4);residual[id]=err[j]/(norm[j]?norm[j]:1);if(!std::isfinite(residual[id]))throw std::runtime_error("Nonfinite residual");for(double v:weights[id])if(!std::isfinite(v))throw std::runtime_error("Nonfinite weight");}
  }
 }
 double compute=elapsed(begin);blas(cublasDestroy(handle));
 std::ofstream out(argv[2]);out<<std::setprecision(17);for(int i=0;i<count;i++){out<<i<<" "<<tasks[i].nodes.size()<<" none "<<residual[i];for(double v:weights[i])out<<" "<<v;out<<"\n";}if(!out)throw std::runtime_error("Output write failed");
 std::cout<<std::setprecision(9)<<"{\"device\":\""<<prop.name<<"\",\"context_seconds\":"<<init<<",\"compute_seconds\":"<<compute<<",\"h2d_seconds\":"<<h2d<<",\"assembly_seconds\":"<<assembly<<",\"solve_seconds\":"<<solve<<",\"d2h_seconds\":"<<d2h<<",\"groups\":"<<groups.size()<<",\"batches\":"<<batches<<"}\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
