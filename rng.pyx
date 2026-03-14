from libc.stdint cimport uint64_t
import numpy as np
cimport numpy as cnp

# rotation
cdef inline uint64_t rotl(uint64_t x, int k):
    return (x << k) | (x >> (64 - k))

cpdef double random_number(uint64_t[::1] s):

    # magic numbers from the paper
    cdef int R = 23
    cdef int A = 17
    cdef int B = 45

    # scrambler (xoshiro256++ logic)
    cdef uint64_t result_int = rotl(s[0] + s[3], R) + s[0]

    # engine
    cdef uint64_t t = s[1] << A

    s[2] ^= s[0]
    s[3] ^= s[1]
    s[1] ^= s[2]
    s[0] ^= s[3]

    s[2] ^= t

    s[3] = rotl(s[3], B)

    # return the float calculation
    return (result_int >> 11) * (1.0 / 9007199254740992.0) # 0 to 1
