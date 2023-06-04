# Matching & Social Influence (OLA Project 056984)
Consider a setting in which a company wants to increase the visibility of its products by using social influence techniques to reach possible customers. Then, it has to match the reached customers with some of its products.


<details>
  <summary><h2>Environment</h2></summary>
  <br>
  We assume that a round corresponds to one day. The network of customers is organized in a graph defined by:
<ul>
<li>a set of 30 customers;</li>
<li>a set of edges connecting the customers. These edges describe the influence among the customers; we assume that only ~10% (50) of the possible edges are present;</li>
<li>each edge has a possibly different activation probability;</li>
<li>for each user, two binary features can be observed by the company,  call them F1 and F2; </li>
<li>customers can be of three different classes according to these features, call them C1, C2, C3;</li> 
<li>these three classes differ in terms of the reward of matching the customer with the items;</li>
<li>at each round, the company can choose three seeds to activate in the social network.</li>
</ul>
<br>
Moreover, the company has three classes of products D1, D2, and D3, and:
<ul>
<li>for every product of type Dj and class of customer Ci, specify a reward distribution F(Dj, Ci) of matching the product “j” with the customer “i”;</li>
<li>each reward distribution is a Gaussian distribution;</li>
<li>
for every product of type Dj, specify the number of units of this product; each type of product has 3 units;</li>
<li>
each unit of product can be matched only with one customer, and each customer can be matched with a single product.</li>
</ul>

The time horizon to use in the experiments is 365 rounds long. At each round, after the set of seeds is selected, the information cascade and matching is repeated sufficiently many times.

 </details>



<details>
  <summary><h2>Clairvoyant optimization algorithm</h2></summary>
  Consider the case in which the company can directly observe the type of each customer Ci. The objective function to maximize is defined as the sum of the expected reward of the couples of matched customers to products. In particular, for each influenced customer of type Ci matched with a product of type Di the reward is the expected value of the distribution F(Dj, Ci).

The optimization algorithm that we suggest is divided into two steps:
<ol>
<li>Find the node that, when it is a seed, gives the highest marginal increase in the number of total activated nodes. Then, fix it as a seed and find the one among the remaining ones that, when added, gives the highest increase. Repeat the same procedure also for the last node. This is called the <b>greedy algorithm</b>. When looking for the nodes that give the highest increase in the number of total activated nodes, simulate the social influence process by using a <b>Monte Carlo technique</b> with a sufficiently large number of runs.</li>
<li>
When the optimal set of seeds is fixed, compute the value of the optimum by simulating multiple runs of the social influence process and, for each set of activated nodes, <b>compute the value of the optimal matching</b>. The value of the optimum is computed as an expectation over these runs. If there are more activated users than products, define an opportune number of dummy items such that the total number of items equals the number of users. There is no reward when a user of any class is matched with a dummy item. The case in which there are more items than users can be handled in a similar way.
</li>
</ol>
</details>


<details>
  <summary><h2>Requirements</h2></summary>
    <ul>
      <li><details>
        <summary><h3>Step 0: Motivations and environment design</h3></summary>
        Imagine and motivate a realistic application fitting with the scenario above. Describe all the parameters needed to build the simulator.
        </details></li>
      <li>
      <details>
        <summary> <h3>Step 1: Learning for social influence</h3>
        </summary>
        Assume that all the properties of the graph are known except for the edge activation probabilities. Apply the greedy algorithm to the problem of maximizing the expected number of activated customers, where each edge activation probability is replaced with its upper confidence bound (in a <b>UCB1-like fashion</b>). Furthermore, apply the <b>greedy algorithm</b> to the same problem when estimating edge activation probabilities with Beta distributions and sampling is used (in a <b>TS-like fashion</b>). Report the plots with the average (over a sufficiently large number of runs) value and standard deviation of the cumulative regret, cumulative reward, instantaneous regret, and instantaneous reward.
        </details>
      </li>
      <li>
      <details>
        <summary>
        <h3>Step 2: Learning for matching</h3> 
        </summary>
        Consider the case in which the company can observe the type of each customer Ci. Moreover, assume that the set of seeds is fixed to the optimal solution found when the activation probabilities are known. On the other hand, suppose that the reward distributions F(Dj, Ci) for the matching are unknown. Apply an <b>upper confidence bound matching algorithm</b>in which the value of a matching is substituted with its upper confidence bound. Do the same using a <b>TS-like algorithm</b>. Report the plots with the average (over a sufficiently large number of runs) value and standard deviation of the cumulative regret, cumulative reward, instantaneous regret, and instantaneous reward.
        </details>
      </li>
      <li>
      <details>
        <summary>
          <h3>Step 3: Learning for joint social influence and matching</h3>
        </summary>
        Consider the case in which the company can observe the type of each customer Ci.  Moreover, assume that both the edge activation probabilities and reward distributions F(Dj, Ci) are unknown. Apply jointly the <b>greedy algorithm</b> (for influence maximization) and the matching algorithm using upper confidence bound in place of the edge activation probabilities and the expected reward of each match.  Apply jointly the greedy algorithm (for influence maximization) and the <b>matching algorithm using the TS algorithm</b> to estimate the edge activation probabilities and the expected reward of each match. Report the plots of the average value and standard deviation of the cumulative regret, cumulative reward, instantaneous regret, and instantaneous reward.
        </details>
      </li>
      <li>
      <details>
        <summary>
        <h3>Step 4: Contexts and their generation</h3>
        </summary>
        Consider the case in which the company cannot observe the type of each customer Ci, but only the features F1 and F2. Moreover, no information about the edge activation probabilities and the reward distributions F(Dj, Ci) is known beforehand. The <b>structure of the contexts is not known beforehand and needs to be learned from data</b>. Important remark: the learner does not know how many contexts there are, while it can only observe the features and data associated with the features. <b>Apply the UCB and TS algorithms (as in Step 3) paired with a context generation algorithm</b>, reporting the plots with the average (over a sufficiently large number of runs) value and standard deviation of the cumulative regret, cumulative reward, instantaneous regret, and instantaneous reward. Apply the context generation algorithms every two weeks of the simulation. Compare the performance of the designed algorithm with the one in Step 3 (that can observe the context).
        </details>
        </li>
        <li>
        <details>
        <summary>
          <h3>Step 5: Dealing with non-stationary environments with two abrupt changes</h3>
        </summary>
                        Assume that all the properties of the graph are known except for the edge activation probabilities. Assume that the edge activation probabilities are non-stationary, being <b>subject to seasonal phases (3 different phases spread over 365 days).</b> Provide motivation for the phases. Apply the <b>greedy algorithm</b> to the problem of maximizing the expected number of activated customers, where each edge activation probability is replaced with its <b>upper confidence bound (in a UCB1-like fashion)</b>. Moreover, apply two non-stationary flavors of the algorithm. The <b>first one is passive and exploits a sliding window</b>, while the second one is active and exploits a <b>change detection test</b>. Provide a sensitivity analysis of the algorithms, evaluating different values of the length of the sliding window in the first case and different values for the parameters of the change detection test in the second case. Report the plots of the average value and standard deviation of the cumulative regret, cumulative reward, instantaneous regret, and instantaneous reward.
        </details>
      </li>
      <li>
      <details>
        <summary>
          <h3>Step 6: Dealing with non-stationary environments with many abrupt changes</h3>
        </summary>
        Develop the EXP3 algorithm, which is devoted to dealing with adversarial settings. This algorithm is also used to deal with non-stationary settings when no information about the specific form of non-stationarity is known beforehand. Consider a simplified version of Step 5  in which the company chooses a single seed to activate in the social network at each round. First, apply the EXP3 algorithm and the algorithms designed in Step 5 to this simplified version of the setting. The expected result is that EXP3 performs much worse than the two non-stationary versions of UCB1. Subsequently, consider a different non-stationary setting with a higher non-stationarity degree. Such a degree can be modeled by having a large number of phases that frequently change. In particular, consider 5 phases, each one associated with a different optimal price, and these phases cyclically change with a high frequency. In this new setting, apply EXP3, UCB1, and the two non-stationary flavors of UBC1. The expected result is that EXP3 outperforms the non-stationary flavors of UCB1.
        </details>
      </li>
    </ul>
    </details>
</details>

<h2>TO DO</h2>
<ul>
<li>Preliminary - Environment & clairvoyant algorithm
<ol>
<li>Simulate social influence process using Montecarlo technique</li>
<li>Implement greedy algorithm for influence maximisation</li>
<li>Find optimal matching</li>
</ol>
</li>
<li>Step 1 - Learning for social influence
<ol>
<li>Implement UCB1-like algorithm for estimating activation probabilities</li>
<li>Implement TS-like algorithm for estimating activation probabilities</li>
<li>Implement greedy algorithm for influence maximisation</li>
</ol>
</li>
<li>Step 2 - Learning for matching</li>
<ol>
<li>Implement UCB1-like algorithm for matching</li>
<li>Implement TS-like algorithm for matching</li>
<ol>


</ol>
</li>
<li>Step 3 - Learning for joint social influence and matching</li>
<ol>
<li>Implement greedy algorithm for influence maximisation</li>
<li>Implement TS-like algorithm for matching</li>
</ol>
</li>
<li>Step 4 - Contexts and their generation</li>
<ol>
<li>Implement UCB1-like algorithm for influence maximisation</li>