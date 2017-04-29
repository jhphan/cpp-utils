#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_INIT([MIBLab], 0.9, [jhphan@gmail.com])
AM_INIT_AUTOMAKE([MIBLab], 0.9)
AC_CONFIG_SRCDIR([src/checkerboard.cpp])
AC_CONFIG_FILES([Makefile include/Makefile libsvm-2.8/Makefile liblr/Makefile lib/Makefile src/Makefile])

# Checks for programs.
AC_PROG_RANLIB
AC_PROG_CXX
AC_PROG_CC
#AC_PROG_LIBTOOL

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([stdlib.h string.h sys/time.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST
AC_HEADER_TIME

# Checks for library functions.
AC_CHECK_FUNCS([gettimeofday sqrt sysinfo])

OPENCV=""
AC_ARG_WITH(opencv,
AC_HELP_STRING([--with-opencv=/path/to/opencv], [Location of OpenCV library]),
 [OPENCV="$withval"])
AC_SUBST(OPENCV)
AM_CONDITIONAL(HAS_OPENCV, [test -n "$OPENCV"])

LAPACK=""
AC_ARG_WITH(lapack,
AC_HELP_STRING([--with-lapack=/path/to/lapack], [Location of LAPACK library]),
 [LAPACK="$withval"])
AC_SUBST(LAPACK)
AM_CONDITIONAL(HAS_LAPACK, [test -n "$LAPACK"])

BOOST=""
AC_ARG_WITH(boost,
AC_HELP_STRING([--with-boost=/path/to/boost], [Location of boost library]),
 [BOOST="$withval"])
AC_SUBST(BOOST)
AM_CONDITIONAL(HAS_BOOST, [test -n "$BOOST"])

TSNNLS=""
AC_ARG_WITH(tsnnls,
AC_HELP_STRING([--with-tsnnls=/path/to/tsnnls], [Location of tsnnls library]),
 [TSNNLS="$withval"])
AC_SUBST(TSNNLS)
AM_CONDITIONAL(HAS_TSNNLS, [test -n "$TSNNLS"])

AC_OUTPUT
