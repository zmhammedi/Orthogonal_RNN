/*  Author: Zakaria Mhammedi
The University of Melbourne and Data61 (2016 - 2017) */

#include "Python.h"
#include "numpy/arrayobject.h"
#include "C_fun.h"
#include <math.h>

/* ==== Set up the methods table ====================== */
static PyMethodDef C_funMethods[] = {
	{"F", F, METH_VARARGS, NULL},
	{"GradF", GradF, METH_VARARGS, NULL},
	{NULL, NULL, 0, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the functions ====================== */
static struct PyModuleDef C_fun =
{
    PyModuleDef_HEAD_INIT,
    "C_fun", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    C_funMethods
};

PyMODINIT_FUNC PyInit_C_fun(void)  {
    import_array();  // Must be present for NumPy.  Called first after above line.
	return PyModule_Create(&C_fun);
}

/* ==== Local Forward Propagation   ========= */
static PyObject *F(PyObject *self, PyObject *args)
{
    PyArrayObject *pyU, *pyh, *pyc, *pyh_tilde;
    double *U, *h, *c, *h_tilde;
    int i,j,n,m, dims[2];

    /* Parse tuples separately since args will differ between C fcns */
    if (!PyArg_ParseTuple(args, "OOOO", &pyU, &pyh, &pyc, &pyh_tilde))
         return NULL;
    if (NULL == pyU || NULL == pyh || NULL == pyc || NULL == pyh_tilde)
         return NULL;

    /* Get the dimensions of the input */
    n=dims[0]=pyU->dimensions[0];
    m=dims[1]=pyU->dimensions[1];

    /* Change contiguous arrays into C ** arrays (Memory is Allocated!) */
    U=(double *) pyU->data;
    h=(double *) pyh->data;
    c=(double *) pyc->data;
    h_tilde=(double *) pyh_tilde->data;

    for (i=0; i<n; i++)
        c[i] = h[i];

    /* Calculating h_tilde and h=H(:,1)  */
    for (j=0; j<m; j++){
        for (i=m-j-1; i<n; i++)
            h_tilde[m-j-1] += U[i*m+m-j-1] * c[i]; 
        h_tilde[m-j-1] *= 2.;  /* Here it is assumed that the columns of U (the reflection vectors) have unit norms. The normalization is done during SGD (see sgd.py). */
        for (i=0; i<n; i++)
            c[i] -= h_tilde[m-j-1] * U[i*m+m-j-1];
    }
    Py_RETURN_NONE;
}


/* ==== Square matrix components function & multiply by int and float ========= */
static PyObject *GradF(PyObject *self, PyObject *args)
{
    PyArrayObject *pyU, *pyG, *pyh, *pyg, *pyBPg, *pyh_tilde, *pyc_tilde;
    double *U, *G, *h, *g, *BPg, *h_tilde, *c_tilde;
    int i,j,n,m, dims[2];

    /* Parse tuples separately since args will differ between C fcns */
    if (!PyArg_ParseTuple(args, "OOOOOOO", &pyU, &pyh, &pyBPg, &pyG, &pyg, &pyh_tilde, &pyc_tilde))
        return NULL;
    if (NULL == pyU || NULL == pyh || NULL == pyBPg || NULL == pyG || NULL == pyg || NULL == pyh_tilde ||
        NULL == pyc_tilde)
        return NULL;

    /* Get the dimensions of the input */
    n=dims[0]=pyU->dimensions[0];
    m=dims[1]=pyU->dimensions[1];

    U=(double *) pyU->data;
    G=(double *) pyG->data;
    h=(double *) pyh->data;
    g=(double *) pyg->data;
    BPg=(double *) pyBPg->data;
    h_tilde=(double *) pyh_tilde->data;
    c_tilde=(double *) pyc_tilde->data;
    double* H = malloc((size_t)  n * (m + 1) * sizeof(double));

    /* Initialisation */
    for (i=0; i<n; i++){
        H[i*(m+1)+m] = h[i];
        g[i] = BPg[i];
    }

    /* Calculating G and g  */
    for (j=0; j<m; j++){
        for (i=m-j-1; i<n; i++)
            h_tilde[m-j-1] += U[i*m+m-j-1] * H[i*(m+1)+m-j]; 
        h_tilde[m-j-1] *= 2.;
        for (i=0; i<n; i++)
            H[i*(m+1)+m-j-1] = H[i*(m+1)+m-j] - h_tilde[m-j-1] * U[i*m+m-j-1];
    }
    for (j=0; j<m; j++){
        for (i=j; i<n; i++)
            c_tilde[j] += U[i*m+j] * g[i]; 
        c_tilde[j] *= 2.;
        for (i=j; i<n; i++)
            g[i] -= c_tilde[j] * U[i*m+j];
        for (i=0; i<n; i++){
            G[i*m+j]= -h_tilde[j]*g[i] - c_tilde[j] * H[i*(m+1)+j+1];
        }
     }

    /* Free memory, close file and return */
    free(H);
    Py_RETURN_NONE;
}
