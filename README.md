# sae-informed-perturbations

Algorithm: 
1. Determine most important words in text input using word saliency score from Meng and Wattenhofer
2. Find SAE "concept vectors" with which most important words are most strongly associated
3. For each MIW, find synonym which is *least* strongly associated with that SAE concept vector
4. Replace