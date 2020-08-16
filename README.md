# Must-read papers on Recommender System

[![wechat](https://img.shields.io/badge/wechat-ML--RSer-blue)](https://mp.weixin.qq.com/mp/qrcode?scene=10000004&size=102&__biz=MzA4NTUxNTE4Ng==&mid=2247483655&idx=1&sn=5ed421a66f03a31fbab722192b8ccae2&send_time=) [![update](https://img.shields.io/badge/update-weekly-blueviolet)](#must-read-papers-on-recommender-system) [![license](https://img.shields.io/github/license/hongleizhang/RSPapers)](https://github.com/hongleizhang/RSPapers/blob/master/LICENSE)

This repository provides a list of papers including comprehensive surveys, classical recommender system, social recommender system, deep learing-based recommender system, cold start problem in recommender system, hashing for recommender system, exploration and exploitation problem, explainability in recommender system  as well as  click through rate prediction for recommender system. For more posts about recommender systems, please transfer to [ML_RSer](https://mp.weixin.qq.com/s/WqpRxKBUHYBeuTh6AETpTQ).


- [**New!**] Add the new part of [**Conversational RS**](https://github.com/hongleizhang/RSPapers/tree/master/13-Conversational_RS).

- [**New!**] Add the new part of [**Review based RS**](https://github.com/hongleizhang/RSPapers/tree/master/12-Review_RS).

- [**New!**] Add the new part of [**Knowledge Graph for RS**](https://github.com/hongleizhang/RSPapers/tree/master/11-Knowledge_Graph_for_RS).

============================================================================

**01-Surveys:** a set of comprehensive surveys about recommender system, such as hybrid recommender systems, social recommender systems, poi recommender systems, deep-learning based recommonder systems and so on.

**02-General RS:** a set of famous recommendation papers which make predictions with some classic models and practical theory.

**03-Social RS:** several papers which utilize trust/social information in order to alleviate the sparsity of ratings data.

**04-Deep Learning-based RS:** a set of papers to build a recommender system with deep learning techniques.

**05-Cold Start Problem in RS:** some papers specifically dealing with the cold start problems inherent in collaborative filtering.

**06-POI RS:** it focus on helping users explore attractive locations with the information of location-based social networks.

**07-Efficient RS:** some hashing techniques for recommender system in order to training and making recommendation efficiently.

**08-EE Problem in RS:** some articles about exploration and exploitation problems in recommendation.

**09-Explainability on RS:** it focus on addressing the problem of 'why', they not only provide
the user with the recommendations, but also make the user aware why such items are recommended by generating recommendation explanations.

**10-CTR Prediction for RS:** as one part of recommendation, click-through rate prediction focuses on the elaboration of candidate sets for recommendation.

**11-Knowledge Graph for RS:** knowledge graph, as the side information of behavior interaction matrix in recent years, which can effectively alleviate the problem of data sparsity and cold start, and can provide a reliable explanation for recommendation results.

**12-Review_RS:** some articles about review or text based recommendations.

**13-Conversational_RS:** Make use of natural language processing technology to interactively provide recommendations

============================================================================

\*All papers are sorted by year for clarity.

## Surveys

* Burke et al. **Hybrid Recommender Systems: Survey and Experiments.** USER MODEL USER-ADAP, 2002.

* Adomavicius et al. **Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions.** IEEE TKDE, 2005.

* Su et al. **A survey of collaborative filtering techniques.** Advances in artificial intelligence, 2009.

* Asela et al. **A Survey of Accuracy Evaluation Metrics of Recommendation Tasks.** J. Mach. Learn. Res, 2009.

* Cacheda et al. **Comparison of collaborative filtering algorithms: Limitations of current techniques and proposals for scalable, high-performance recommender systems.** ACM TWEB, 2011.

* Zhang et al. **Tag-aware recommender systems: a state-of-the-art survey.** J COMPUT SCI TECHNOL, 2011.

* Tang et al. **Social recommendation: a review.** SNAM, 2013.

* Yang et al. **A survey of collaborative filtering based social recommender systems.** COMPUT COMMUN, 2014.

* Shi et al. **Collaborative filtering beyond the user-item matrix: A survey of the state of the art and future challenges.** ACM COMPUT SURV, 2014.

* Chen et al. **Recommender systems based on user reviews: the state of the art.** USER MODEL USER-ADAP, 2015.

* Xu et al. **Social networking meets recommender systems: survey.** Int.J.Social Network Mining, 2015.

* Yu et al. **A survey of point-of-interest recommendation in location-based social networks.** In Workshops at AAAI, 2015.

* Efthalia et al. **Parallel and Distributed Collaborative Filtering: A Survey.** Comput. Surv., 2016.

* Singhal et al. **Use of Deep Learning in Modern Recommendation System: A Summary of Recent Works.** arXiv, 2017.

* Muhammad et al. **Cross Domain Recommender Systems: A Systematic Literature Review.** ACM Comput. Surv, 2017.

* Massimo et al. **Sequence-Aware Recommender Systems.** ACM Comput. Surv, 2018.

* Zhang et al. **Deep learning based recommender system: A survey and new perspectives.** ACM Comput.Surv, 2018.

* Batmaz et al. **A review on deep learning for recommender systems: challenges and remedies.** Artificial Intelligence Review, 2018.

* Zhang et al. **Explainable Recommendation: A Survey and New Perspectives.** arXiv, 2018.

* Liu et al. **Survey of matrix factorization based recommendation methods by integrating social information.** Journal of Software, 2018.

* Shoujin et al. **A Survey on Session-based Recommender Systems.** arXiv, 2019.

* Shoujin et al. **Sequential Recommender Systems: Challenges, Progress and Prospects.** IJCAI, 2019.

* Zhu et al. **Research Commentary on Recommendations with Side Information: A Survey and Research Directions.**  Electron. Commer. Res. Appl., 2019.

* Lina et al. **Recommendations on the Internet of Things: Requirements, Challenges, and Directions.** IEEE Internet Comput., 2019.

* Sriharsha et al. **A Survey on Group Recommender Systems.** J. Intell. Inf. Syst., 2019.

* Dietmar et al. **A Survey on Conversational Recommender Systems.** arXiv, 2020.

* Qingyu et al. **A Survey on Knowledge Graph-Based Recommender Systems.** arXiv, 2020.

* Yang et al. **Deep Learning on Knowledge Graph for Recommender System: A Survey.** arXiv, 2020.

* Wang et al. **Graph Learning Approaches to Recommender Systems: A Review.** arXiv, 2020.

* Yashar et al. **Adversarial Machine Learning in Recommender Systems-State of the art and Challenges.** arXiv, 2020.

* May et al. **Recommender Systems for the Internet of Things:A Survey.** arXiv, 2020.



## General Recommender System

* Goldberg et al. **Using collaborative filtering to weave an information tapestry.** COMMUN ACM, 1992.

* Resnick et al. **GroupLens: an open architecture for collaborative filtering of netnews.** CSCW, 1994.

* Sarwar et al. **Application of dimensionality reduction in recommender system-a case study.** 2000.

* Sarwar et al. **Item-based collaborative filtering recommendation algorithms.** WWW, 2001.

* Linden et al. **Amazon.com recommendations: Item-to-item collaborative filtering.** IEEE INTERNET COMPUT, 2003.

* Lemire et al. **Slope one predictors for online rating-based collaborative filtering.** SDM, 2005.

* Zhou et al. **Bipartite network projection and personal recommendation.** Physical Review E, 2007.

* Mnih et al. **Probabilistic matrix factorization.** NIPS, 2008.

* Koren et al. **Factorization meets the neighborhood: a multifaceted collaborative filtering model.** SIGKDD, 2008.

* Pan et al. **One-class collaborative filtering.** ICDM, 2008.

* Hu et al. **Collaborative filtering for implicit feedback datasets.** ICDM, 2008.

* Weimer et al. **Improving maximum margin matrix factorization.** Machine Learning, 2008.

* Koren et al. **Matrix factorization techniques for recommender systems.** Computer, 2009.

* Agarwal et al. **Regression-based latent factor models.** SIGKDD, 2009.

* Koren et al. **The bellkor solution to the netflix grand prize.** Netflix prize documentation, 2009.

* Rendle et al. **BPR: Bayesian personalized ranking from implicit feedback.** UAI, 2009.

* Koren et al. **Collaborative filtering with temporal dynamics.** COMMUN ACM, 2010.

* Khoshneshin et al. **Collaborative filtering via euclidean embedding.** RecSys, 2010.

* Liu et al. **Online evolutionary collaborative filtering Recsys.** RecSys, 2010.

* Koren et al. **Factor in the neighbors: Scalable and accurate collaborative filtering.** TKDD, 2010.

* Chen et al. **Feature-based matrix factorization.** arXiv, 2011.

* Rendle. **Learning recommender systems with adaptive regularization.** WSDM, 2012.

* Zhong et al. **Contextual collaborative filtering via hierarchical matrix factorization.** SDM, 2012.

* Lee et al. **Local low-rank matrix approximation.** ICML, 2013.

* Kabbur et al. **Fism: factored item similarity models for top-n recommender systems.** KDD, 2013.

* Johnson et al. **Logistic Matrix Factorization for Implicit Feedback Data.** NIPS Workshop, 2014.

* Hu et al. **Your neighbors affect your ratings: on geographical neighborhood influence to rating prediction.** SIGIR, 2014.

* Hernández-Lobato et al. **Probabilistic matrix factorization with non-random missing data.** ICML, 2014.

* Yang et al. **TopicMF: Simultaneously Exploiting Ratings and Reviews for Recommendation.** AAAI, 2014.

* Shi et al. **Semantic path based personalized recommendation on weighted heterogeneous information networks.** CIKM, 2015.

* Grbovic et al. **E-commerce in your inbox: Product recommendations at scale.** KDD, 2015.

* Barkan et al. **Item2vec: neural item embedding for collaborative filtering.** Machine Learning for Signal Processing, 2016.

* Liang et al. **Modeling user exposure in recommendation.** WWW, 2016.

* He et al. **Fast matrix factorization for online recommendation with implicit feedback.** SIGIR, 2016.

* Hsieh et al. **Collaborative metric learning.** WWW, 2017.

* He et al. **Translation-based Recommendation.** RecSys, 2017.

* Bayeret al. **A generic coordinate descent framework for learning from implicit feedback.** WWW, 2017.

* Ruining et al. **Translation-based Recommendation.** RecSys, 2017.

* Rajiv et al. **Translation-based factorization machines for sequential recommendation.** RecSys 2018.

* Gao et al. **BiNE: Bipartite Network Embedding.** SIGIR, 2018.

* Xiangnan et al. **Adversarial Personalized Ranking for Recommendation.** SIGIR, 2018.

* Zhang et al. **Metric Factorization: Recommendation beyond Matrix Factorization.** 2018.

* Lei et al. **Spectral Collaborative Filtering.** RecSys, 2018.

* Feng et al. **Adversarial Collaborative Neural Network for Robust Recommendation.** SIGIR, 2019.

* Chen et al. **Collaborative Similarity Embedding for Recommender Systems.** arXiv, 2019.

* Chuan et al. **Heterogeneous Information Network Embedding for Recommendation.** TKDE, 2019.

* Huafeng et al. **Deep Generative Ranking for Personalized Recommendation.** Recsys, 2019.

* Xiang et al. **Neural Graph Collaborative Filtering.** SIGIR, 2019.

* Wenjie et al. **Denoising Implicit Feedback for Recommendation.** arXiv, 2020.

* Rendle et al. **Neural Collaborative Filtering vs. Matrix Factorization Revisited.** arXiv, 2020.


## Social Recommender System

* Ma, Hao, et al. **Sorec: social recommendation using probabilistic matrix factorization.** CIKM, 2008.

* Jamali et al. **Trustwalker: a random walk model for combining trust-based and item-based recommendation.** SIGKDD, 2009.

* Ma et al. **Learning to recommend with trust and distrust relationships.** RecSys, 2009.

* Ma et al. **Learning to recommend with social trust ensemble.** SIGIR, 2009.

* Jamali et al. **A matrix factorization technique with trust propagation for recommendation in social networks.** RecSys, 2010.

* Ma, Hao, et al. **Recommender systems with social regularization.** WSDM, 2011.

* Ma, Hao et al. **Learning to recommend with explicit and implicit social relations.** ACM T INTEL SYST TEC, 2011.

* Ma, Hao. **An experimental study on implicit social recommendation.** SIGIR, 2013.

* Yang et al. **Social collaborative filtering by trust.** IJCAI, 2013.

* Jiliang et al. **Exploiting Local and Global Social Context for Recommendation.** IJCAI, 2013.

* Zhao et al. **Leveraging social connections to improve personalized ranking for collaborative filtering.** CIKM, 2014.

* Chen et al. **Context-aware collaborative topic regression with social matrix factorization for recommender systems.** AAAI, 2014.

* Guo et al. **TrustSVD: Collaborative Filtering with Both the Explicit and Implicit Influence of User Trust and of Item Ratings.** AAAI, 2015.

* Wang et al. **Social recommendation with strong and weak ties.** CIKM, 2016.

* Jiliang et al. **Recommendation with Social Dimensions.** AAAI, 2016.

* Li et al. **Social recommendation using Euclidean embedding.** IJCNN, 2017.

* Zhang et al. **Collaborative User Network Embedding for Social Recommender Systems.** SDM, 2017.

* Yang et al. **Social collaborative filtering by trust.** IEEE T PATTERN ANAL, 2017.

* Park et al. **UniWalk: Explainable and Accurate Recommendation for Rating and Network Data.** arXiv, 2017.

* Rafailidis et al. **Learning to Rank with Trust and Distrust in Recommender Systems.** RecSys, 2017.

* Xixi et al. **Additive Co-Clustering with Social Influence for Recommendation.** RecSys, 2017.

* Zhao et al. **Collaborative Filtering with Social Local Models.** ICDM, 2017.

* Wang et al. **Collaborative Filtering with Social Exposure: A Modular Approach to Social Recommendation.** AAAI, 2018.

* Wenqi et al. **Deep Modeling of Social Relations for Recommendation.** AAAI, 2018

* Xuying et al. **Personalized Privacy-Preserving Social Recommendation.** AAAI,2018.

* Wen et al. **Network embedding based recommendation method in social networks.** WWW Poster, 2018.

* Lin et al. **Recommender Systems with Characterized Social Regularization.** CIKM Short Paper, 2018.

* Yu et al. **Adaptive implicit friends identification over heterogeneous network for social recommendation.** CIKM, 2018.

* Honglei et al. **Social Collaborative Filtering Ensemble.** PRICAI, 2018.

* Wenqi et al. **Graph Neural Networks for Social Recommendation.** WWW, 2019.

* Song et al. **Session-based Social Recommendation via Dynamic Graph Attention Networks.** WSDM, 2019.

* Wenqi et al. **Deep Social Collaborative Filtering.** RecSys, 2019.

* Wenqi et al. **Deep Adversarial Social Recommendation.** IJCAI, 2019.

* Qitian et al. **Feature Evolution Based Multi-Task Learning for Collaborative Filtering with Social Trust.** IJCAI, 2019.

* Wu et al. **SocialGCN: An Efficient Graph Convolutional Network based Model for Social Recommendation.** AAAI, 2019.

* Wu et al. **Dual Graph Attention Networks for Deep Latent Representation of Multifaceted Social Effects in Recommender System.** WWW, 2019.

* Chong Chen et al. **An Efficient Adaptive Transfer Neural Network for Social-aware Recommendation.** SIGIR, 2019.

* Wu et al. **A Neural Influence Diffusion Model for Social Recommendation.** SIGIR, 2019.

* Cheng et al. **An Efficient Adaptive Transfer Neural Network for Social-aware Recommendation.** SIGIR, 2019.

* Yang et al. **Modelling High-Order Social Relations for Item Recommendation.** arXiv, 2020.

* Junliang et al. **Enhance Social Recommendation with Adversarial Graph Convolutional Networks.** TKDE, 2020.

* Chaochao et al. **Secure Social Recommendation based on Secret Sharing.** arXiv, 2020.


## Deep Learning based Recommender System

* Salakhutdinov et al. **Restricted Boltzmann machines for collaborative filtering.** ICML, 2007.

* Aäron et al. **Deep content-based music recommendation.** NIPS, 2013.

* Huang et al. **Learning deep structured semantic models for web search using clickthrough data.** CIKM, 2013.

* Wang et al. **Collaborative deep learning for recommender systems.** KDD, 2015.

* Sedhain et al. **Autorec: Autoencoders meet collaborative filtering.** WWW, 2015.

* Ali et al. **A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems.** WWW, 2015

* Li et al. **Deep collaborative filtering via marginalized denoising auto-encoder.** CIKM, 2015.

* Ruining et al. **VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback.** AAAI, 2016.

* Hidasi et al. **Session-based recommendations with recurrent neural networks.** ICLR, 2016.

* Covington et al. **Deep neural networks for youtube recommendations.** RecSys, 2016.

* Cheng et al. **Wide & deep learning for recommender systems.** Workshop on RecSys, 2016.

* Zheng et al. **A neural autoregressive approach to collaborative filtering.** ICML, 2016.

* Wu et al. **Collaborative denoising auto-encoders for top-n recommender systems.** WSDM, 2016.

* Tan et al. **Improved recurrent neural networks for session-based recommendations.** Workshop on Deep Learning for Recommender Systems, 2016.

* Kang et al. **Visually-Aware Fashion Recommendation and Design with Generative Image Models.** ICDM, 2017.

* Wu et al. **Recurrent Recommender Networks.** WSDM, 2017.

* Lian et al. **CCCFNet: a content-boosted collaborative filtering neural network for cross domain recommender systems.** WWW, 2017.

* He et al. **Neural collaborative filtering.** WWW, 2017.

* Zhao et al. **Leveraging Long and Short-term Information in Content-aware Movie Recommendation.** arXiv, 2017.

* Li et al. **Deep Collaborative Autoencoder for Recommender Systems: A Unified Framework for Explicit and Implicit Feedback.** arXiv, 2017.

* Xue et al. **Deep Matrix Factorization Models for Recommender Systems.** IJCAI, 2017. [code](https://github.com/RuidongZ/Deep_Matrix_Factorization_Models)

* He et al. **Outer Product-based Neural Collaborative Filtering.** IJCAI, 2018.

* Dong et al. **CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.** CIKM, 2018.

* Zhao et al. **Learning and Transferring IDs Representation in E-commerce.** KDD, 2018.

* Liang et al. **Variational Autoencoders for Collaborative Filtering.** WWW, 2018.

* Ebesu et al. **Collaborative Memory Network for Recommendation Systems.** SIGIR, 2018.

* Lian et al. **xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems.** KDD, 2018.

* Zhang et al. **Next Item Recommendation with Self-Attention.** 2018.

* Li et al. **Learning from History and Present: Next-item Recommendation via Discriminatively Exploiting User Behaviors.** KDD, 2018.

* Grbovic et al. **Real-time Personalization using Embeddings for Search Ranking at Airbnb.** KDD, 2018.

* Ying et al. **Graph Convolutional Neural Networks for Web-Scale Recommender Systems.** KDD, 2018.

* Hu et al. **Leveraging meta-path based context for top-n recommendation with a neural co-attention model.** KDD, 2018.

* Christakopoulou et al. **Local Latent Space Models for Top-N Recommendation.** KDD, 2018.

* Bhagat et al. **Buy It Again: Modeling Repeat Purchase Recommendations.** KDD, 2018.

* Wang et al. **Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba.** KDD, 2018.

* Tran et al. **Regularizing Matrix Factorization with User and Item Embeddings for Recommendation.** CIKM, 2018.

* Zhou et al. **Micro behaviors: A new perspective in e-commerce recommender systems.** WSDM, 2018.

* Chen et al. **Sequential recommendation with user memory networks.** WSDM, 2018.

* Beutel et al. **Latent Cross: Making Use of Context in Recurrent Recommender Systems.** WSDM, 2018.

* Tang et al. **Personalized top-n sequential recommendation via convolutional sequence embedding.** WSDM, 2018.

* Chae et al. **CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.** CIKM, 2018.

* Wu et al. **Session-based Recommendation with Graph Neural Networks.** AAAI, 2019.

* Zhi-Hong et al. **DeepCF: A Unified Framework of Representation Learning and Matching Function Learning in Recommender System.** AAAI, 2019.

* Zeping et al. **Adaptive User Modeling with Long and Short-Term Preferences for Personalized Recommendation.** IJCAI, 2019.

* Dong Xi et al. **BPAM: Recommendation Based on BP Neural Network with Attention Mechanism.** IJCAI, 2019.

* Xin et al. **CFM: Convolutional Factorization Machines for Context-Aware Recommendation.** IJCAI, 2019.

* Xiao Zhou et al. **Collaborative Metric Learning with Memory Network for Multi-Relational Recommender Systems.** IJCAI, 2019.

* Junyang et al. **Convolutional Gaussian Embeddings for Personalized Recommendation with Uncertainty.** IJCAI, 2019.

* Feng Yuan et al. **DARec: Deep Domain Adaptation for Cross-Domain Recommendation via Transferring Rating Patterns.** IJCAI, 2019.

* Yanan et al. **Learning Shared Vertex Representation in Heterogeneous Graphs with Convolutional Networks for Recommendation.** IJCAI, 2019.

* Jiani et al. **STAR-GCN: Stacked and Reconstructed Graph Convolutional Networks for Recommender Systems.** IJCAI, 2019.

* An et al. **CosRec: 2D Convolutional Neural Networks for Sequential Recommendation.** CIKM, 2019.

* Sun et al. **BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer.** CIKM, 2019.

* Hongwei et al. **Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation.** WWW, 2019.

* Maurizio et al. **Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches.** RecSys, 2019.

* Maurizio et al. **A Troubling Analysis of Reproducibility and Progress in Recommender Systems Research.** arXiv, 2019.

* Xin et al. **CFM: Convolutional factorization machines for context-aware recommendation.** IJCAI, 2019.

* Huafeng et al. **Deep Global and Local Generative Model for Recommendation.** WWW, 2020.


## Cold Start Problem in Recommender System

* Schein et al. **Methods and metrics for cold-start recommendations.** SIGIR, 2002.

* Seung-Taek et al. **Pairwise Preference Regression for Cold-start Recommendation.** RecSys, 2009.

* Gantner et al. **Learning attribute-to-feature mappings for cold-start recommendations.** ICDM, 2010.

* Sedhain et al. **Social collaborative filtering for cold-start recommendations.** RecSys, 2014.

* Zhang et al. **Addressing cold start in recommender systems: A semi-supervised co-training algorithm.** SIGIR, 2014.

* Kula. **Metadata embeddings for user and item cold-start recommendations.** arXiv, 2015.

* Sedhain et al. **Low-Rank Linear Cold-Start Recommendation from Social Data.** AAAI. 2017.

* Man et al. **Cross-domain recommendation: an embedding and mapping approach.** IJCAI, 2017.

* Cohen et al. **Expediting Exploration by Attribute-to-Feature Mapping for Cold-Start Recommendations.** RecSys, 2017.

* Dureddy et al. **Handling Cold-Start Collaborative Filtering with Reinforcement Learning.** arXiv, 2018.

* Fu et al. **Deeply Fusing Reviews and Contents for Cold Start Users in Cross-Domain Recommendation Systems.** AAAI, 2019.

* Li. **From Zero-Shot Learning to Cold-Start Recommendation.** AAAI, 2019

* Hoyeop. **Estimating Personalized Preferences Through Meta-Learning for User Cold-Start Recommendation.** KDD, 2019.

* Ruobing et al. **Internal and Contextual Attention Network for Cold-start Multi-channel Matching in Recommendation.** IJCAI, 2020.

* Sun et al. **LARA: Attribute-to-feature Adversarial Learning for New-item Recommendation.** WSDM, 2020.

* Lu et al. **Meta-learning on heterogeneous information networks for cold-start recommendation.** KDD, 2020.


## POI Recommender System

* Mao et al. **Exploiting geographical influence for collaborative point-of-interest recommendation.** SIGIR, 2011.

* Chen et al. **Fused matrix factorization with geographical and social influence in location-based social networks.** AAAI, 2012.

* Jia et al. **iGSLR: personalized geo-social location recommen dation: a kernel density estimation approach.** SIGSPA, 2013.

* Jia et al. **Lore: exploiting sequential influence for location recommendations.** SIGSPATIAL, 2014

* Jia et al. **Geosoca: Exploiting geographical, social and cat egorical correlations for point-of-interest recommendations.** SIGIR, 2015.

* Huayu et al. **Point-of-Interest Recommendations:Learning Potential Check-ins from Friends.** KDD, 2016.

* Jing et al. **Category-aware next point-of-interest recommendation via listwise Bayesian personalized ranking.** IJCAI, 2017.

* Jarana et al. **A Personalised Ranking Framework with Multiple Sampling Criteria for Venue Recommendation.** CIKM, 2017.

* Huayu et al. **Learning user's intrinsic and extrinsic interests for point-of-interest recommendation: a unified approach.** IJCAI, 2017.

* Feng et al. **POI2Vec: Geographical Latent Representation for Predicting Future Visitors.** AAAI, 2017.

* Wei Liu et al. **Geo-ALM: POI Recommendation by Fusing Geographical Information and Adversarial Learning Mechanism.** IJCAI, 2019.

* et al. **Contextualized Point-of-Interest Recommendation.** IJCAI, 2020.


## Efficient RS

* Karatzoglou et al. **Collaborative filtering on a budget.** AISTAT, 2010.

* Zhou et al. **Learning binary codes for collaborative filtering.** SIGKDD, 2012.

* Zhang et al. **Preference preserving hashing for efficient recommendation.** SIGIR, 2014.

* Zhang et al. **Discrete collaborative filtering.** SIGIR, 2016.

* Lian et al. **Discrete Content-aware Matrix Factorization.** SIGKDD, 2017.

* Han et al. **Discrete Factorization Machines for Fast Feature-based Recommendatio.** IJCAI, 2018.

* Guibing et al. **Discrete Trust-aware Matrix Factorization for Fast Recommendation.** IJCAI, 2019.

* Chenghao et al. **Discrete Social Recommendation.** AAAI, 2019.

* Defu et al. **LightRec: a Memory and Search-Efficient Recommender System.** WWW, 2020.

* Yang et al. **A Generic Network Compression Framework for Sequential Recommender Systems.** SIGIR, 2020.

* Xiangnan et al. **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.** SIGIR, 2020.


## EE in RS

* Auer et al. **Using confidence bounds for exploitation-exploration trade-offs.** JMLR, 2002.

* Li et al. **A contextual-bandit approach to personalized news article recommendation.** WWW, 2010.

* Li et al. **Exploitation and exploration in a performance based contextual advertising system.** SIGKDD, 2010.

* Chapelle et al. **An empirical evaluation of thompson sampling.** NIPS, 2011.

* Féraud et al. **Random forest for the contextual bandit problem.** Artificial Intelligence and Statistics. 2016.

* Li et al. **Collaborative filtering bandits.** SIGIR, 2016.

* Wang et al. **Factorization Bandits for Interactive Recommendation.** AAAI, 2017.



## Explainability on RS

* Park et al. **UniWalk: Explainable and Accurate Recommendation for Rating and Network Data.** arXiv, 2017.

* Huang et al. **Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks.** SIGIR, 2018.

* Wang et al. **Tem: Tree-enhanced embedding model for explainable recommendation.** WWW, 2018.

* Lu et al. **Why I like it: multi-task learning for recommendation and explanation.** RecSys, 2018.

* Wang et al. **Explainable Reasoning over Knowledge Graphs for Recommendation.** AAAI, 2019.

* Cao et al. **Unifying Knowledge Graph Learning and Recommendation: Towards a Better Understanding of User Preferences.** WWW, 2019.

* Zhongxia et al. **Co-Attentive Multi-Task Learning for Explainable Recommendation.** IJCAI, 2019.

* Min et al. **Explainable Fashion Recommendation: A Semantic Attribute Region Guided Approach.** IJCAI, 2019.

* Peijie et al. **Dual Learning for Explainable Recommendation: Towards Unifying User Preference Prediction and Review Generation.** WWW, 2020.


## CTR Prediction for RS

* Richardson et al. **Predicting Clicks - Estimating the Click-Through Rate for New Ads.** WWW, 2007.

* Steffen et al. **Fast Context-aware Recommendations with Factorization Machines.** SIGIR, 2011.

* H. Brendan McMahan et al. **Ad Click Prediction a View from the Trenches.** KDD, 2013.

* Aäron et al. **Deep content-based music recommendation.** NIPS, 2013.

* Xinran He et al. **Practical Lessons from Predicting Clicks on Ads at Facebook.** ADKDD, 2014.

* Ying Shan et al. **Deep Crossing - Web-Scale Modeling without Manually Crafted Combinatorial Features.** KDD, 2016.

* Weinan Zhang et al. **Deep Learning over Multi-field Categorical Data.** arXiv, 2016.

* Yuchin et al. **Field-aware Factorization Machines for CTR Prediction.** RecSys, 2016.

* Cheng et al. **Wide & Deep Learning for Recommender Systems.** arXiv, 2016.

* Qu et al. **Product-based neural networks for user response prediction.** ICDM, 2016.

* Chao et al. **Recurrent recommender networks.** WSDM, 2017.

* Guo et al. **A Factorization-Machine based Neural Network for CTR Prediction.** arXiv, 2017.

* Xiao et al. **Attentional Factorization Machines - Learning the Weight of Feature Interactions via Attention Networks.** arXiv, 2017.

* Guo et al. **Deepfm: A factorization-machine based neural network for ctr prediction.** IJCAI, 2017

* Gai et al. **Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction.** arXiv, 2017.

* Xiangnan He et al. **Neural Factorization Machines for Sparse Predictive Analytics.** arXiv, 2017.

* Ruoxi et al. **Deep & Cross Network for Ad Click Predictions.** ADKDD, 2017

* Zhou et al. **Deep Interest Network for Click-Through Rate Prediction.** KDD 2018.

* Lian et al. **xDeepFM - Combining Explicit and Implicit Feature Interactions for Recommender Systems.** arXiv, 2018.

* Zhou et al **Deep Session Interest Network for Click-Through Rate Prediction.** IJCAI, 2019.

* Zhou et al. **Deep Interest Evolution Network for Click-Through Rate Prediction.** AAAI, 2019 

* Yang et al. **Operation-aware Neural Networks for User Response Prediction.** 2019.

* Liu et al. **Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction.** 2019.

* Wentao et al. **Deep Spatio-Temporal Neural Networks for Click-Through Rate Prediction.** KDD, 2019.

* Qi et al. **Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction.** KDD, 2019.

* Fuzheng et al. **Collaborative Knowledge Base Embedding for Recommender Systems.** KDD, 2016.

* Shu et al. **TFNet: Multi-Semantic Feature Interaction for CTR Prediction.** SIGIR, 2020.

* Weinan et al. **Deep Interest with Hierarchical Attention Network for Click-Through Rate Prediction.** SIGIR, 2020



## Knowledge Graph for RS

* Fuzheng et al. **Collaborative Knowledge Base Embedding for Recommender Systems.** KDD, 2016.

* Hongwei et al. **DKN: Deep Knowledge-Aware Network for News Recommendation.** WWW, 2018.

* Hongwei et al. **Ripplenet-Propagating user preferences on the knowledge graph for recommender systems.** CIKM, 2018.

* Hongwei et al. **Knowledge-aware Graph Neural Networks with Label Smoothness Regularization for Recommender Systems.** KDD, 2019.

* Hongwei et al. **Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation.** WWW, 2019.

* Xiang et al. **Reinforced Negative Sampling over Knowledge Graph for Recommendation.** WWW, 2020.


## Review based RS

* Chong et al. **Collaborative Topic Modeling for Recommending Scientific Articles.** KDD, 2011.

* McAuley et al. **Hidden Factors and Hidden Topics: Understanding Rating Dimensions with Review Text.** RecSys, 2013.

* Guang et al. **Ratings Meet Reviews, a Combined Approach to Recommend.** RecSys, 2014

* Wei et al. **Collaborative Multi-Level Embedding Learning from Reviews for Rating Prediction.** IJCAI, 2016.

* Kim et al. **Convolutional Matrix Factorization for Document Context-Aware Recommendation.** RecSys, 2016.

* Yunzhi et al. **Rating-boosted latent topics Understanding users and items with ratings and reviews.** IJCAI, 2016.

* Seo et al. **Interpretable Convolutional Neural Networks with Dual Local and Global Attention for Review Rating Prediction.** RecSys, 2016.

* Lei et al. **Joint Deep Modeling of Users and Items Using Reviews for Recommendation.** WSDM, 2017.

* Zhiyong et al. **A3NCF An Adaptive Aspect Attention Model for Rating Prediction.**  IJCAI, 2018.

* Jinyao et al. **ANR: Aspect-based Neural Recommender** CIKM, 2018.

* Yichao et al.  **Coevolutionary recommendation model Mutual learning between ratings and reviews.** WWW, 2018.

* Yi et al. **Multi-pointer co-attention networks for recommendation.** KDD, 2018.

* Chong et al. **Neural attentional rating regression with review-level explanations.** WWW, 2018.

* Libing et al. **Parl: Let strangers speak out what you like.** CIKM, 2018.

* Libing et al. **A context-aware user-item representation learning for item recommendation.** TOIS, 2019.

* Donghua et al. **DAML: Dual Attention Mutual Learning between Ratings and Reviews for Item Recommendation.** KDD, 2019.

* Liu et al. **NRPA Neural Recommendation with Personalized Attention.** SIGIR, 2019.

* Noveen et al. **How Useful are Reviews for Recommendation? A Critical Review and Potential Improvements.** SIGIR, 2020.


## Conversational RS

* Zhao et al. **Interactive collaborative filtering.** CIKM, 2013.

* Negar et al. **Context adaptation in interactive recommender systems.** RecSys, 2014.

* Yasser et al. **History-guided conversational recommendation.** WWW, 2014.

* Konstantina et al. **Towards Conversational Recommender Systems.** KDD, 2016.

* Konstantina et al. **Q&R: A Two-Stage Approach toward Interactive Recommendation.** KDD, 2018.

* Sun et al. **Conversational Recommender System.** SIGIR, 2018.

* Yongfeng et al. **Towards Conversational Search and Recommendation: System Ask, User Respond.** CIKM, 2018.

* Raymond et al. **Towards Deep Conversational Recommendations.** NeurIPS, 2018.

* Tong et al. **A Visual Dialog Augmented Interactive Recommender System.** KDD, 2019.

* Qibin et al. **Towards Knowledge-Based Recommender Dialog System.** EMNLP, 2019.

* Yuanjiang et al. **Adversarial Attacks and Detection on Reinforcement Learning-Based Interactive Recommender Systems.** SIGIR, 2020.

* Wenqiang et al. **Conversational Recommendation: Formulation, Methods, and Evaluation.** SIGIR, 2020.

* Xingshan et al. **Dynamic Online Conversation Recommendation.** ACL, 2020.

* Wenqiang et al. **Estimation-Action-Reflection: Towards Deep Interaction Between Conversational and Recommender Systems.** WSDM, 2020.

* Kun et al. **Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion.** KDD, 2020.

* Wenqiang et al. **Interactive Path Reasoning on Graph for Conversational Recommendation.** KDD, 2020.

* Sijin et al. **Interactive Recommender System via Knowledge Graph-enhanced Reinforcement Learning.** SIGIR, 2020.

* Kai et al. **Latent Linear Critiquing for Conversational Recommender Systems.** WWW, 2020.

* Lixin et al. **Neural Interactive Collaborative Filtering.** SIGIR, 2020.

* Lixin et al. **Pseudo Dyna-Q: A Reinforcement Learning Framework for Interactive Recommendation.** WSDM, 2020.

* Shijun et al. **Seamlessly Unifying Attributes and Items: Conversational Recommendation for Cold-Start Users.** arXiv, 2020.

* Zeming et al. **Towards Conversational Recommendation over Multi-Type Dialogs.** ACL, 2020.

* Zhongxia et al. **Towards Explainable Conversational Recommendation.** IJCAI, 2020.

* Jie et al. **Towards Question-based Recommender Systems.** SIGIR, 2020.

* Hu et al. **User Memory Reasoning for Conversational Recommendation.** arXiv, 2020.

* Kai et al. **Latent Linear Critiquing for Conversational Recommender Systems.** WWW, 2020.


## RSAlgorithms

Recently, we have launched an open source project [**RSAlgorithms**](https://github.com/hongleizhang/RSAlgorithms), which provides an integrated training and testing framework. In this framework, we implement a set of classical **traditional recommendation methods** which make predictions only using rating data and **social recommendation methods** which utilize trust/social information in order to alleviate the sparsity of ratings data. Besides, we have collected some classical methods implemented by others for your convenience.


## Acknowledgements

Specially summerize the papers about Recommender Systems for you, and if you have any questions, please contact me generously. Last but not least, the ability of myself is limited so I sincerely look forward to working with you to contribute it.


Thank @**ShawnSu** for collecting papers about POI Recommender Systems.

Thank @**Wang Zhe** for his advice about EE in RS.

Highly thank @**Yujia Zhang** for her summary on Hashing for RS.

Thank @**Zixuan Yang** and @**vicki1109** for his collecting papers about CTR Prediction for RS.

Thank @**ShomyLiu** for collecting papers about Review based RS.

Specially appreciate Professor @[**Jun Wu**](http://faculty.bjtu.edu.cn/8620/) for his attentive guidance in my research career.

WeChat Official Account: [ML-RSer](https://mp.weixin.qq.com/mp/qrcode?scene=10000004&size=102&__biz=MzA4NTUxNTE4Ng==&mid=2247483655&idx=1&sn=5ed421a66f03a31fbab722192b8ccae2&send_time=)

My ZhiHu: [Honglei Zhang](https://www.zhihu.com/people/hongleizhang)





