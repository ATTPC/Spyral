# interpolate Module

This module contains submodules related to interpolation. These submodules are all JIT-compatible. It was found that the scipy interpolation solutions were all too slow for use in Spyral. As such, we developed interpolation schemes compatible with the Numba Just-In-Time compiler. To be JIT-compatible, these interpolaters are not very generic, and as such should not be used without carefully considering if they fit the use case. The submodules are

- [bilinear](bilinear.md)
- [linear](linear.md)