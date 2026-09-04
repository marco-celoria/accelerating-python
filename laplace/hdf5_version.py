import h5py
print("h5py version:", h5py.__version__)
print("MPI-enabled:", h5py.get_config().mpi)
print("HDF5 version:", h5py.version.hdf5_version)
