# Must-read papers on Recommender System

This repository provides a list of papers including **surveys**, **classical recommender system**, **social recommender system**, **deep learing-based recommender system**, **cold start problem in recommender system**, **Hashing for RS** and **EE in RS**. 

It contains *a set of comprehensive surveys* about recommender system, collaborative filtering, social recommender system and deep learning based recommender system, *a set of famous recommendation papers* which make predictions with some classic models and *social recommendation papers* which utilize trust/social information in order to alleviate the sparsity of ratings data, and *a set of papers to build a recommender system with deep learning techniques*. Next, there are also *some papers specifically dealing with the cold start problems inherent in collaborative filtering*. Then, we collect some *hashing techniques for recommender system* in order to training efficiently. Last, there are also some articles about *exploration and exploitation problems* in recommendation.

The order of papers is sorted by year.

## Surveys

* Burke et al. **Hybrid Recommender Systems: Survey and Experiments**[J]. User Modeling and User-Adapted Interaction, 2002, 12(4):331-370.

* Adomavicius et al. **Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions**. IEEE transactions on knowledge and data engineering 17.6 (2005): 734-749.

* Su et al. **A survey of collaborative filtering techniques.** Advances in artificial intelligence 2009 (2009): 4.

* Koren et al. **Matrix factorization techniques for recommender systems**. Computer 42.8 (2009).

* Cacheda et al. **Comparison of collaborative filtering algorithms: Limitations of current techniques and proposals for scalable, high-performance recommender systems**. ACM Transactions on the Web (TWEB) 5.1 (2011): 2.

* Tang et al. **Social recommendation: a review**. Social Network Analysis and Mining 3.4 (2013): 1113-1133.

* Yang et al. **A survey of collaborative filtering based social recommender systems**. Computer Communications 41 (2014): 1-10.

* Shi et al. **Collaborative filtering beyond the user-item matrix: A survey of the state of the art and future challenges**. ACM Computing Surveys (CSUR) 47.1 (2014): 3.

* Chen et al. **Recommender systems based on user reviews: the state of the art.** User Modeling and User-Adapted Interaction 25.2 (2015): 99-154.

* Yu et al. **A survey of point-of-interest recommendation in location-based social networks.** In Workshops at AAAI, 2015.

* Zhang et al. **Deep learning based recommender system: A survey and new perspectives**. arXiv preprint arXiv:1707.07435 (2017).


## Classical Recommender System

* Goldberg et al. **Using collaborative filtering to weave an information tapestry.** Communications of the ACM 35.12 (1992): 61-70.

* Resnick et al. **GroupLens: an open architecture for collaborative filtering of netnews**. Proceedings of the 1994 ACM conference on Computer supported cooperative work. ACM, 1994.

* Sarwar et al. **Item-based collaborative filtering recommendation algorithms**. WWW, 2001.

* Zhou et al. **Bipartite network projection and personal recommendation**. Physical Review E 76.4 (2007): 046115.

* Mnih et al. **Probabilistic matrix factorization**. NIPS (2008): 1257-1264.

* Koren et al. **Factorization meets the neighborhood: a multifaceted collaborative filtering model**. SIGKDD, 2008.

* Pan et al. **One-class collaborative filtering.** ICDM, 2008.

* Hu et al. **Collaborative filtering for implicit feedback datasets**. ICDM, 2008.

* Weimer et al. **Improving maximum margin matrix factorization**. Machine Learning 72.3 (2008): 263-276.

* Koren et al. **The bellkor solution to the netflix grand prize.** Netflix prize documentation 81 (2009): 1-10.

* Rendle et al. **BPR: Bayesian personalized ranking from implicit feedback**. UAI, 2009.

* Koren et al. **Collaborative filtering with temporal dynamics.** Communications of the ACM 53.4 (2010): 89-97.

* Khoshneshin et al. **Collaborative filtering via euclidean embedding**. RecSys, 2010.

* Koren et al. **Factor in the neighbors: Scalable and accurate collaborative filtering**. ACM Transactions on Knowledge Discovery from Data (TKDD) 4.1 (2010): 1.

* Zhong et al. **Contextual collaborative filtering via hierarchical matrix factorization.** SDM, 2012.

* Lee et al. **Local low-rank matrix approximation**. ICML. 2013.

* Hu et al. **Your neighbors affect your ratings: on geographical neighborhood influence to rating prediction**. SIGIR, 2014.

* Hernández-Lobato et al. **Probabilistic matrix factorization with non-random missing data.** ICML. 2014.

* Grbovic et al. **E-commerce in your inbox: Product recommendations at scale**. ICKM, 2015.

* Barkan et al. **Item2vec: neural item embedding for collaborative filtering**. Machine Learning for Signal Processing (MLSP), 2016.

* Liang et al. **Modeling user exposure in recommendation.** WWW, 2016.

* Hsieh et al. **Collaborative metric learning**. WWW, 2017.



## Social Recommender System

* Ma, Hao, et al. **Sorec: social recommendation using probabilistic matrix factorization**. Proceedings of the 17th ACM conference on Information and knowledge management. ACM, 2008.

* Jamali et al. **Trustwalker: a random walk model for combining trust-based and item-based recommendation**. SIGKDD, 2009.

* Ma et al. **Learning to recommend with trust and distrust relationships**. RecSys, 2009.

* Ma et al. **Learning to recommend with social trust ensemble**. SIGIR, 2009.

* Jamali et al. **A matrix factorization technique with trust propagation for recommendation in social networks**. RecSys, 2010.

* Ma, Hao, et al. **Recommender systems with social regularization**. WSDM, 2011.

* Ma, Hao et al. **Learning to recommend with explicit and implicit social relations.** ACM Transactions on Intelligent Systems and Technology (TIST) 2.3 (2011): 29.

* Ma, Hao. **An experimental study on implicit social recommendation**. SIGIR, 2013.

* Zhao et al. **Leveraging social connections to improve personalized ranking for collaborative filtering**. ICKM, 2014.

* Chen et al. **Context-aware collaborative topic regression with social matrix factorization for recommender systems**. AAAI, 2014.

* Guo et al. **TrustSVD: Collaborative Filtering with Both the Explicit and Implicit Influence of User Trust and of Item Ratings**. AAAI, 2015.

* Wang et al. **Social recommendation with strong and weak ties**. ICKM, 2016.

* Li et al. **Social recommendation using Euclidean embedding**. IJCNN, 2017.

* Zhang et al. **Collaborative User Network Embedding for Social Recommender Systems**. SDM, 2017.

* Yang et al. **Social collaborative filtering by trust.** IEEE transactions on pattern analysis and machine intelligence 39.8 (2017): 1633-1647.

* Park et al. **UniWalk: Explainable and Accurate Recommendation for Rating and Network Data**. arXiv preprint arXiv:1710.07134 (2017).

* Rafailidis et al. **Learning to Rank with Trust and Distrust in Recommender Systems**. RecSys, 2017.

* Zhao et al. **Collaborative Filtering with Social Local Models**. ICDM, 2017.

* Wang et al. **Collaborative Filtering with Social Exposure: A Modular Approach to Social Recommendation.** AAAI 2018.



## Deep Learning based Recommender System

* Salakhutdinov et al. **Restricted Boltzmann machines for collaborative filtering**. ICML, 2007.

* Wang et al. **Collaborative deep learning for recommender systems.** ICDM, 2015.

* Sedhain et al. **Autorec: Autoencoders meet collaborative filtering.** WWW, 2015.

* Zheng et al. **A neural autoregressive approach to collaborative filtering.** ICML, 2016.

* Wu et al. **Collaborative denoising auto-encoders for top-n recommender systems.** WSDM, 2016.

* Kim et al. **Convolutional matrix factorization for document context-aware recommendation**. RecSys, 2016.

* He et al. **Neural collaborative filtering.** WWW, 2017.

* Liang et al. **Variational Autoencoders for Collaborative Filtering.** WWW, 2018.


## Cold Start Problem in Recommender System

* Gantner et al. **Learning attribute-to-feature mappings for cold-start recommendations**. Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 2010.

* Sedhain et al. **Low-Rank Linear Cold-Start Recommendation from Social Data**. AAAI. 2017.

* Man et al. **Cross-domain recommendation: an embedding and mapping approach**. IJCAI, 2017.

* Cohen et al. **Expediting Exploration by Attribute-to-Feature Mapping for Cold-Start Recommendations**. RecSys, 2017.


## POI Recommender System

* Mao Ye et al. **Exploiting geographical influence for collaborative point-of-interest recommendation.** SIGIR, 2011.

* Chen Cheng et al. **Fused matrix factorization with geographical and social influence in location-based social networks.** AAAI, 2012.

* Jia-Dong et al. **iGSLR: personalized geo-social location recommen dation: a kernel density estimation approach.** SIGSPA, 2013.

* Jia-Dong et al. **Lore: exploiting sequential influence for location recommendations.** SIGSPATIAL, 2014

* Jia-Dong et al. **Geosoca: Exploiting geographical, social and cat egorical correlations for point-of-interest recommendations.** SIGIR, 2015.

* Huayu et al. **Point-of-Interest Recommendations:Learning Potential Check-ins from Friends.** KDD, 2016.

* Jing et al. **Category-aware next point-of-interest recommendation via listwise Bayesian personalized ranking.** IJCAI,2017.

* Jarana et al. **A Personalised Ranking Framework with Multiple Sampling Criteria for Venue Recommendation.** CIKM,2017.

* Huayu et al. **Learning user's intrinsic and extrinsic interests for point-of-interest recommendation: a unified approach.** IJCAI, 2017.


## Hashing for RS

In progress...


## EE in RS

* Auer et al. **Using confidence bounds for exploitation-exploration trade-offs.** Journal of Machine Learning Research 3.Nov (2002): 397-422.

* Li et al. **A contextual-bandit approach to personalized news article recommendation.** WWW, 2010.

* Li et al. **Exploitation and exploration in a performance based contextual advertising system.** SIGKDD, 2010.

* Chapelle et al. **An empirical evaluation of thompson sampling.** NIPS, 2011.

* Féraud et al. **Random forest for the contextual bandit problem.** Artificial Intelligence and Statistics. 2016.


## Acknowledgements

Specially summerize the papers about Recommender Systems for you, and if you have any questions, please contact me generously. Last but not least, the ability of myself is limited so I sincerely look forward to working with you to contribute it.

Greatly thank @**ShawnSu** for collecting papers about POI Recommender Systems.

Greatly thank @**Wang Zhe** for his advice about EE in RS.

My Homepage: [Honglei Zhang](http://midas.bjtu.edu.cn/Home/MemberStudent/27)

My ZhiHu: [Honglei Zhang](https://www.zhihu.com/people/hongleizhang)

My Gmail: hongleizhang1993@gmail.com


