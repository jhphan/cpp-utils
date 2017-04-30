#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_INIT([MIBLab], 0.9, [jhphan@gmail.com])
AM_INIT_AUTOMAKE([MIBLab], 0.9)
AC_CONFIG_SRCDIR([src/extract_rows.cpp])
AC_CONFIG_FILES([Makefile include/Makefile lib/Makefile src/Makefile])

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

AC_OUTPUT
