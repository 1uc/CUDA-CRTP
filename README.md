# CUDA-CRTP
Subtle CRTP run-time error in CUDA.

Compile with

    make

This will create two binaries `with-user-cast` and `no-user-cast`. The
difference is only whether the CRTP base has a user defined cast operator.

The device throws a "Invalid instruction" error with this cast, but not without.
