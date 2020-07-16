# Total cumulative mutual information (NOMAD Analytics Toolkit)

The identification of relevant features, i.e., the driving variables that determine a process or the property of a system, is an essential part of the analysis of data sets whose entries are described by a large number of variables. The preferred measure for quantifying the relevance of nonlinear statistical dependencies is mutual information, which requires as input probability distributions. Probability distributions cannot be reliably sampled and estimated from limited data, especially for real-valued data samples such as lengths or energies.

This interactive notebook includes the original implementation of total cumulative mutual information (TCMI) to reproduce the main results presented in the publication:

> B. Regler, M. Scheffler, and L. M. Ghiringhelli: "TCMI: a non-parametric mutual-dependence estimator for multivariate continuous distributions"  [<a href="https://arxiv.org/abs/2001.11212">arxiv:2001.11212</a>] [<a href="https://arxiv.org/pdf/2001.11212">pdf</a>]

TCMI is a measure of the relevance of mutual dependencies based on cumulative probability distributions. TCMI can be estimated directly from sample data and is a non-parametric, robust and deterministic measure that facilitates comparisons and rankings between feature sets with different cardinality. The ranking induced by TCMI allows for feature selection, i.e. the identification of the set of relevant features that are statistical related to the process or the property of a system, while taking into account the number of data samples as well as the cardinality of the feature subsets.

It is compared to [Cumulative mutual information (CMI)](https://dx.doi.org/10.1137/1.9781611972832.22), [Multivariate maximal correlation analysis (MAC)](http://proceedings.mlr.press/v32/nguyenc14.html), [Universal dependency analysis (UDS)](https://dx.doi.org/10.1137/1.9781611974348.89), and [Monte Carlo dependency estimation (MCDE)](https://dx.doi.org/10.1145/3335783.3335795).

This repository (notebook and code) is released under the [Apache License, Version 2.0](http://www.apache.org/licenses/). Please see the [LICENSE](LICENSE) file.

**Important notes:**
<ul style="color: #8b0000; font-style: italic;">
<li>All comparisons have been computed with the Java package <code>MCDE</code> written in Scala, which is not part of the repository. To use the most recent and maintained implementation, please visit <a href="https://github.com/edouardfouche/MCDE">https://github.com/edouardfouche/MCDE</a> and run all examples with 50,000 iterations.</li>
<li>For the sake of simplicity, all results have been cached. However, results can be recalculated after adjusting the respective test sections. Depending on the test, the calculation time ranges from minutes to days.</li>
</ul>

#### Project Information:

**Maintainer:** <a href="https://github.com/sommerregen" style="color: #808080;" title="Maintainer">&#x1F464; Benjamin Regler</a>

**Status:** <span style="color: #008000;">&#10004; Under active development</span>

## License

Copyright (c) 2018+ Fritz Haber Institute of the Max Planck Society ([Benjamin Regler][github]).

[github]: https://github.com/sommerregen "GitLab account of Benjamin Regler"
