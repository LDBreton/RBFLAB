window.MathJax = {
  loader: {load: ["[tex]/boldsymbol"]},
  tex: {
    packages: {"[+]": ["boldsymbol"]},
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};
// Full-page navigation uses MathJax startup typesetting.
