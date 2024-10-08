import gudhi as gd
import numpy as np

# Extracts persistence diagrams from a simplex tree 'st'
def diagram_from_simplex_tree(st, mode, dim=0):
    st.compute_persistence(min_persistence=-1.)
    dgm0 = st.persistence_intervals_in_dimension(0)[:, 1]   
    # Extracts the death times of the 0-dimensional persistence intervals from the computed persistence.

    if mode == "superlevel":
        dgm0 = - dgm0[np.where(np.isfinite(dgm0))]
    elif mode == "sublevel":
        dgm0 = dgm0[np.where(np.isfinite(dgm0))]
    if dim==0:
        return dgm0
    elif dim==1:
        dgm1 = st.persistence_intervals_in_dimension(1)[:,0]
        return dgm0, dgm1
    # If dim is 0, returns the 0-dimensional persistence intervals (dgm0)
    # If dim is 1, additionally extracts and returns the birth times of the 1-dimensional persistence intervals (dgm1)

    # track how clusters (connected components) and loops change over a filtration
    # calculate persistence of various features in the simplicial complex.


def sum_diag_from_point_cloud(X, mode="superlevel"):
    rc = gd.RipsComplex(points=X)
    st = rc.create_simplex_tree(max_dimension=1)
    dgm = diagram_from_simplex_tree(st, mode=mode)
    sum_dgm = np.sum(dgm)
    return sum_dgm

# This function leverages the earlier function to compute a persistence diagram from a
# given point cloud and sums all the values in the diagram.
# The result is a single scalar value that gquantitatively summarizes the
# topological features captured in the persistence diagram.

