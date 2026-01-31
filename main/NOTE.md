# Improve dataset
- C/C++ repo with cwe-ids
# Replay improve
## Long term mem implement
1.  Replace retrieval method:
- history store & retrieve + promting (soft/hard): add promting for history remind
2.  history processing (keep replay but process context)
    
    -> Semantic Redundant Information Filtering -> remove Semantic duplicate samples
    
    -> sumaziation -> reduce size of sample by using only vulnerability code lines in stead of function (data process or using llm) -> just for remind

    -> replay priority: clustering data history and statistic type cluster for each task -> get vulnerability frequency over time