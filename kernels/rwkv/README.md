# RWKV device kernels

New RWKV-specific AIE C++ kernels belong here. This directory intentionally
contains no placeholder implementation that could be mistaken for a working
RWKV engine.

Keep host inference/state management in C++, and add matching offline design
builders under `tools/compile/`. Define the model generation, dimensions,
weight/storage layout, numerical reference and error tolerance before adding a
recurrent-state kernel. Reuse the validated upstream projection and elementwise
kernels where their layouts and numerical contracts fit.
