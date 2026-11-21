# Traits and concepts tests

Run the suites with Python 3 and NVRTC. The default run compiles all tests to PTX;
it requires neither a GPU nor a working CUDA driver.

```bash
python experimental/libnvcxx/tests/run.py
```

Specify an NVRTC shared library explicitly, or repeat `--nvrtc` to test multiple
versions in one run:

```bash
python experimental/libnvcxx/tests/run.py \
  --nvrtc /path/to/cuda-12.9/lib64/libnvrtc.so \
  --nvrtc /path/to/cuda-13.3/lib64/libnvrtc.so
```

Keep each NVRTC library beside its matching `libnvrtc-builtins` library. If the
loader cannot find the builtins, add that directory to `LD_LIBRARY_PATH`. Each test
is compiled separately with C++20, 128-bit integers, and device functions as the
default execution space. Native runs use `--minimal`. Every native test runs both
with and without `NVCXX_AS_STD`.

## Coverage

- `arithmetic_results.cu`: mixed-type integer rounding, math overload return types,
  duration arithmetic and conversion, and time-point arithmetic.
- `classification_matrix.cu`: fundamental, arithmetic, scalar, compound, integral,
  floating, signedness, pointer, enum, function, object, reference, and cv traits;
  arithmetic concepts. Tests expand across cv and reference combinations for
  built-in types, 128-bit integers, enums, classes, unions, incomplete classes,
  arrays, functions, and member pointers.
- `construction_matrix.cu`: default/copy/move construction and assignment,
  generic construction/assignment, destruction, triviality, and trivial copying.
  Covers deleted, inaccessible, explicit, ref-qualified, and throwing operations.
- `object_matrix.cu` and `object_concepts.cu`: conversion, construction,
  initialization, movable, and copyable concepts and assignment traits, including
  cases where a trait is true but the corresponding concept is false.
- `common_types.cu`: common-type deduction, decay, variadic deduction, user
  specializations, substitution failures, and duration/time-point specializations.
- `transformations.cu`: cv/ref removal, decay, signed/unsigned transformations,
  `declval` result types, constants, conditional selection, SFINAE, disjunction
  short-circuiting, inheritance, and conversion traits.
- `constraint_overloads.cu` and `integral_concepts.cu`: overload ordering,
  constraint refinement, failed substitution, and arithmetic edge cases.
- `type_traits.cu`: focused regression cases and constant evaluation.
- `runtime.cu`: GPU checks for constant evaluation versus runtime evaluation,
  constrained overload selection, move construction, swapping through moves,
  and assignment values, return references, and source states.

Expected results come from explicit type categories and test fixtures. Tests do
not query construction or triviality traits for incomplete classes.

## Compare with libcudacxx

Pass the CCCL include directory containing `cuda/std/concepts` to compile the
shared cases against libcudacxx as an additional reference run:

```bash
python experimental/libnvcxx/tests/run.py \
  --reference-includes /usr/local/cuda/targets/x86_64-linux/include/cccl
```

The reference uses the selected NVRTC library and the supplied CCCL headers;
these versions are independent. It does not enable `NVCXX_AS_STD` or `--minimal`.
The native-only regression for symmetric `same_as` subsumption is excluded from
reference runs because the installed CCCL snapshot fails that case. Custom swap
overloads are outside this library's supported assumptions and are not exercised.
Object concepts use
assignment traits, so they accept assignment operators returning a value other
than a reference to the left operand; libcudacxx requires that reference result.
The destruction concept checks whether destruction is well-formed and accepts
potentially throwing destructors; libcudacxx requires non-throwing destruction.
The tests cover these intentional differences separately.

## Execute the GPU tests

```bash
python experimental/libnvcxx/tests/run.py --filter runtime.cu --run
```

This requires a working CUDA driver and GPU. The driver must support the PTX
version emitted by the selected NVRTC library. Use `--device` to choose the GPU and
`--architecture` to change the default `compute_80` PTX target. The runner launches
the runtime kernel and checks its failure count through the CUDA Driver API.
An explicitly requested GPU run fails if the driver or device is unavailable.

Filter any compile-time suite with a filename glob:

```bash
python experimental/libnvcxx/tests/run.py --filter '*matrix.cu'
```

The runner prints individual results, continues after compilation failures, and
returns a nonzero exit status if any test fails.
