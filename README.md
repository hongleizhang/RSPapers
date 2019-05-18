# Must-read papers on Recommender System

This repository provides a list of papers including comprehensive surveys, classical recommender system, social recommender system, deep learing-based recommender system, cold start problem in recommender system, hashing for recommender system, exploration and exploitation problem as well as explainability in recommender system.

==============================================================================

**01-Surveys:** a set of comprehensive surveys about recommender system, such as hybrid recommender systems, social recommender systems, poi recommender systems, deep-learning based recommonder systems and so on.

**02-Classical RS:** a set of famous recommendation papers which make predictions with some classic models and practical theory.

**03-Social RS:** several papers which utilize trust/social information in order to alleviate the sparsity of ratings data.

**04-Deep Learning-based RS:** a set of papers to build a recommender system with deep learning techniques.

**05-Cold Start Problem in RS:** some papers specifically dealing with the cold start problems inherent in collaborative filtering.

**06-POI RS:** it focus on helping users explore attractive locations with the information of location-based social networks.

**07-Hashing for RS:** some hashing techniques for recommender system in order to training and making recommendation efficiently.

**08-EE Problem in RS:** some articles about exploration and exploitation problems in recommendation.

**09-Explainability on RS:** it focus on addressing the problem of 'why', they not only provide
the user with the recommendations, but also make the user aware why such items are recommended by generating recommendation explanations.

==============================================================================

\*All papers are sorted by year for clarity.

## Surveys

* Burke et al. **Hybrid Recommender Systems: Survey and Experiments.** USER MODEL USER-ADAP, 2002.

* Adomavicius et al. **Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions.** IEEE TKDE, 2005.

* Su et al. **A survey of collaborative filtering techniques.** Advances in artificial intelligence, 2009.

* Cacheda et al. **Comparison of collaborative filtering algorithms: Limitations of current techniques and proposals for scalable, high-performance recommender systems.** ACM TWEB, 2011.

* Zhang et al. **Tag-aware recommender systems: a state-of-the-art survey.** J COMPUT SCI TECHNOL, 2011.

* Tang et al. **Social recommendation: a review.** SNAM, 2013.

* Yang et al. **A survey of collaborative filtering based social recommender systems.** COMPUT COMMUN, 2014.

* Shi et al. **Collaborative filtering beyond the user-item matrix: A survey of the state of the art and future challenges.** ACM COMPUT SURV, 2014.

* Chen et al. **Recommender systems based on user reviews: the state of the art.** USER MODEL USER-ADAP, 2015.

* Xu et al. **Social networking meets recommender systems: survey.** Int.J.Social Network Mining, 2015.

* Yu et al. **A survey of point-of-interest recommendation in location-based social networks.** In Workshops at AAAI, 2015.

* Singhal et al. **Use of Deep Learning in Modern Recommendation System: A Summary of Recent Works.** arXiv, 2017.

* Zhang et al. **Deep learning based recommender system: A survey and new perspectives.** ACM Comput.Surv, 2018.

* Batmaz et al. **A review on deep learning for recommender systems: challenges and remedies.** Artificial Intelligence Review, 2018.

* Zhang et al. **Explainable Recommendation: A Survey and New Perspectives.** arXiv, 2018.

* Shoujin et al. **A Survey on Session-based Recommender Systems.** arXiv, 2019.


## Classical Recommender System

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

* Hu et al. **Your neighbors affect your ratings: on geographical neighborhood influence to rating prediction.** SIGIR, 2014.

* Hernández-Lobato et al. **Probabilistic matrix factorization with non-random missing data.** ICML, 2014.

* Shi et al. **Semantic path based personalized recommendation on weighted heterogeneous information networks.** CIKM, 2015.

* Grbovic et al. **E-commerce in your inbox: Product recommendations at scale.** KDD, 2015.

* Barkan et al. **Item2vec: neural item embedding for collaborative filtering.** Machine Learning for Signal Processing, 2016.

* Liang et al. **Modeling user exposure in recommendation.** WWW, 2016.

* He et al. **Fast matrix factorization for online recommendation with implicit feedback.** SIGIR, 2016.

* Hsieh et al. **Collaborative metric learning.** WWW, 2017.

* Gao et al. **BiNE: Bipartite Network Embedding.** SIGIR, 2018.

* Zhang et al. **Metric Factorization: Recommendation beyond Matrix Factorization.** 2018.

* Chen et al. **Collaborative Similarity Embedding for Recommender Systems.** arXiv, 2019.

* Chuan et al. **Heterogeneous Information Network Embedding for Recommendation.** TKDE, 2019.



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

* Zhao et al. **Leveraging social connections to improve personalized ranking for collaborative filtering.** CIKM, 2014.

* Chen et al. **Context-aware collaborative topic regression with social matrix factorization for recommender systems.** AAAI, 2014.

* Guo et al. **TrustSVD: Collaborative Filtering with Both the Explicit and Implicit Influence of User Trust and of Item Ratings.** AAAI, 2015.

* Wang et al. **Social recommendation with strong and weak ties.** CIKM, 2016.

* Li et al. **Social recommendation using Euclidean embedding.** IJCNN, 2017.

* Zhang et al. **Collaborative User Network Embedding for Social Recommender Systems.** SDM, 2017.

* Yang et al. **Social collaborative filtering by trust.** IEEE T PATTERN ANAL, 2017.

* Park et al. **UniWalk: Explainable and Accurate Recommendation for Rating and Network Data.** arXiv, 2017.

* Rafailidis et al. **Learning to Rank with Trust and Distrust in Recommender Systems.** RecSys, 2017.

* Zhao et al. **Collaborative Filtering with Social Local Models.** ICDM, 2017.

* Wang et al. **Collaborative Filtering with Social Exposure: A Modular Approach to Social Recommendation.** AAAI, 2018.

* Wen et al. **Network embedding based recommendation method in social networks.** WWW Poster, 2018.

* Lin et al. **Recommender Systems with Characterized Social Regularization.** CIKM Short Paper, 2018.

* Yu et al. **Adaptive implicit friends identification over heterogeneous network for social recommendation.** CIKM, 2018.

* Honglei et al. **Social Collaborative Filtering Ensemble.** PRICAI, 2018.

* Wenqi et al. **Graph Neural Networks for Social Recommendation.** WWW, 2019.

* Songet al. **Session-based Social Recommendation via Dynamic Graph Attention Networks.** WSDM, 2019.


## Deep Learning based Recommender System

* Salakhutdinov et al. **Restricted Boltzmann machines for collaborative filtering.** ICML, 2007.

* Wang et al. **Collaborative deep learning for recommender systems.** KDD, 2015.

* Sedhain et al. **Autorec: Autoencoders meet collaborative filtering.** WWW, 2015.

* Hidasi et al. **Session-based recommendations with recurrent neural networks.** ICLR, 2016.

* Covington et al. **Deep neural networks for youtube recommendations.** RecSys, 2016.

* Cheng et al. **Wide & deep learning for recommender systems.** Workshop on RecSys, 2016.

* Zheng et al. **A neural autoregressive approach to collaborative filtering.** ICML, 2016.

* Wu et al. **Collaborative denoising auto-encoders for top-n recommender systems.** WSDM, 2016.

* Kim et al. **Convolutional matrix factorization for document context-aware recommendation.** RecSys, 2016.

* Tan et al. **Improved recurrent neural networks for session-based recommendations.** Workshop on Deep Learning for Recommender Systems, 2016.

* Lian et al. **CCCFNet: a content-boosted collaborative filtering neural network for cross domain recommender systems.** WWW, 2017.

* He et al. **Neural collaborative filtering.** WWW, 2017.

* Zheng et al. **Joint deep modeling of users and items using reviews for recommendation.** WSDM, 2017.

* Zhao et al. **Leveraging Long and Short-term Information in Content-aware Movie Recommendation.** arXiv, 2017.

* Li et al. **Deep Collaborative Autoencoder for Recommender Systems: A Unified Framework for Explicit and Implicit Feedback.** arXiv, 2017.

* Xue et al. **Deep Matrix Factorization Models for Recommender Systems.** IJCAI. 2017. [code](https://github.com/RuidongZ/Deep_Matrix_Factorization_Models)

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


## Cold Start Problem in Recommender System

* Schein et al. **Methods and metrics for cold-start recommendations.** SIGIR, 2002.

* Gantner et al. **Learning attribute-to-feature mappings for cold-start recommendations.** ICDM, 2010.

* Sedhain et al. **Social collaborative filtering for cold-start recommendations.** RecSys, 2014.

* Zhang et al. **Addressing cold start in recommender systems: A semi-supervised co-training algorithm.** SIGIR, 2014.

* Sedhain et al. **Low-Rank Linear Cold-Start Recommendation from Social Data.** AAAI. 2017.

* Man et al. **Cross-domain recommendation: an embedding and mapping approach.** IJCAI, 2017.

* Cohen et al. **Expediting Exploration by Attribute-to-Feature Mapping for Cold-Start Recommendations.** RecSys, 2017.

* Dureddy et al. **Handling Cold-Start Collaborative Filtering with Reinforcement Learning.** arXiv, 2018.

* Fu et al. **Deeply Fusing Reviews and Contents for Cold Start Users in Cross-Domain Recommendation Systems.** AAAI, 2019.

* Li. **From Zero-Shot Learning to Cold-Start Recommendation.** AAAI, 2019


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


## Hashing for RS

* Karatzoglou et al. **Collaborative filtering on a budget.** AISTAT, 2010.

* Zhou et al. **Learning binary codes for collaborative filtering.** SIGKDD, 2012.

* Zhang et al. **Preference preserving hashing for efficient recommendation.** SIGIR, 2014.

* Zhang et al. **Discrete collaborative filtering.** SIGIR, 2016.

* Lian et al. **Discrete Content-aware Matrix Factorization.** SIGKDD, 2017.


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

## RSAlgorithms

Recently, we have launched an open source project [**RSAlgorithms**](https://github.com/hongleizhang/RSAlgorithms), which provides an integrated training and testing framework. In this framework, we implement a set of classical **traditional recommendation methods** which make predictions only using rating data and **social recommendation methods** which utilize trust/social information in order to alleviate the sparsity of ratings data. Besides, we have collected some classical methods implemented by others for your convenience.


## Acknowledgements

Specially summerize the papers about Recommender Systems for you, and if you have any questions, please contact me generously. Last but not least, the ability of myself is limited so I sincerely look forward to working with you to contribute it.


Thank @**ShawnSu** for collecting papers about POI Recommender Systems.

Thank @**Wang Zhe** for his advice about EE in RS.

Highly thank @**Yujia Zhang** for her summary on Hashing for RS.

Specially appreciate Professor @[**Jun Wu**](http://faculty.bjtu.edu.cn/8620/) for his attentive guidance in my research career.

My ZhiHu: [Honglei Zhang](https://www.zhihu.com/people/hongleizhang)

My Gmail: hongleizhang1993@gmail.com


