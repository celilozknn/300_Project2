from mpi4py import MPI


comm = MPI.COMM_WORLD
print("Hello from rank", comm.Get_rank())