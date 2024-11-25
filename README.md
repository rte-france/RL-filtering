# RL-filtering

This library implements a bid filtering method for European balancing markets that uses reinforcement learning, presented in [1] as "Proposed filtering". Do Nothing and Baseline filtering methods are also implemented. Everything necessary for the code to run is provided, including the data described in the paper. The network used to  generate data is the RTS-GMLC [2], an updated IEEE-96 network.

The Train_RLfiltering file can be used to train a Proposed filtering agent. Main_baseline evaluates a Baseline filtering agent over one year of data, and main_noAction evaluates either a Proposed filtering agent or a Do Nothing agent over the year of data, depending on parameters used.

*References*
[1] Girod, M., Donnot, B., Dussartre, V., Terrier, V., Bourmaud, J. Y., & Perez, Y. (2024). Bid filtering for congestion management in European balancing markets–A reinforcement learning approach. Applied Energy, 361, 122892
[2] Barrows, C., Bloom, A., Ehlen, A., Ikäheimo, J., Jorgenson, J., Krishnamurthy, D., ... & Watson, J. P. (2019). The IEEE reliability test system: A proposed 2019 update. IEEE Transactions on Power Systems, 35(1), 119-127.
