# RL-CC-Project
Masters project on TCP optimization using machine learning

1. Install ns3.36, ns3gym 1.0.2, and tensorflow keras modules. Refer to requirements.txt
2. Go to ns3gym directory opengym/examples/RL-CC-Project where the project has been cloned, build the project(add the project in opengym/examples/CMakeLists.txt) and execute using following command
   ./rl_agent.py
3. sim.cc - defines the point to point network where the default values set for parameters are:
           nLeaf = 1 (number of nodes)
           transport_prot = "TcpRl"  (event based Tcp RL CC)
           bottleneck_bandwidth = "2Mbps"
           bottleneck_delay = "0.01ms"
           access_bandwidth = "10Mbps"
           access_delay = "20ms"
           duration = 60.0 secs
           queue_disc_type = "ns3::PfifoFastQueueDisc"
           recovery = "ns3::TcpClassicRecovery"
           sack = true
