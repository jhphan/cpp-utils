bin_PROGRAMS = 
#checkerboard checkerboard_multi test_feature_selection gauss test_classifier test_classifier2 test_ranksum extract_rows data_combination data_combination_rankprod data_combination_choi data_combination_wang data_combination_mdeds data_combination_mdeds2 data_combination_one test_mrmr

gauss_SOURCES = gauss.cpp
gauss_CFLAGS = -g -O2
gauss_CPPFLAGS = -I$(top_srcdir)/include -I$(OPENCV)/include
gauss_LDFLAGS = -L$(top_srcdir)/lib -lmiblab -L$(top_srcdir)/libsvm-2.8 -lsvm -L$(OPENCV)/lib -lml -lcxcore -lcvaux -lopencv_lapack -lzlib -lrt -lgfortran
gauss_LDADD = $(LAPACK)/lapack_linux.a $(LAPACK)/blas_linux.a $(LAPACK)/tmglib_linux.a
gauss_DEPENDENCIES = $(top_srcdir)/lib/libmiblab.a $(top_srcdir)/libsvm-2.8/libsvm.a

checkerboard_SOURCES = checkerboard.cpp
checkerboard_CFLAGS = -g -O2
checkerboard_CPPFLAGS = -I$(top_srcdir)/include -I$(OPENCV)/include
checkerboard_LDFLAGS = -L$(top_srcdir)/lib -lmiblab -L$(top_srcdir)/libsvm-2.8 -lsvm -L$(OPENCV)/lib -lml -lcxcore -lcvaux -lopencv_lapack -lzlib -lrt -lgfortran
checkerboard_LDADD = $(LAPACK)/lapack_linux.a $(LAPACK)/blas_linux.a $(LAPACK)/tmglib_linux.a
checkerboard_DEPENDENCIES = $(top_srcdir)/lib/libmiblab.a $(top_srcdir)/libsvm-2.8/libsvm.a

checkerboard_multi_SOURCES = checkerboard_multi.cpp
checkerboard_multi_CFLAGS = -g -O2
checkerboard_multi_CPPFLAGS = -I$(top_srcdir)/include -I$(OPENCV)/include
checkerboard_multi_LDFLAGS = -L$(top_srcdir)/lib -lmiblab -L$(top_srcdir)/libsvm-2.8 -lsvm -L$(OPENCV)/lib -lml -lcxcore -lcvaux -lopencv_lapack -lzlib -lrt -lgfortran
checkerboard_multi_DEPENDENCIES = $(top_srcdir)/lib/libmiblab.a $(top_srcdir)/libsvm-2.8/libsvm.a

test_feature_selection_SOURCES = test_feature_selection.cpp
test_feature_selection_CFLAGS = -g -O2
test_feature_selection_CPPFLAGS = -I$(top_srcdir)/include -I$(BOOST)
test_feature_selection_LDFLAGS = -L$(top_srcdir)/lib -lmiblab
test_feature_selection_DEPENDENCIES = $(top_srcdir)/lib/libmiblab.a

test_classifier_SOURCES = test_classifier.cpp
test_classifier_CFLAGS = -g -O2
test_classifier_CPPFLAGS = -I$(top_srcdir)/include -I$(BOOST)
test_classifier_LDFLAGS = -L$(top_srcdir)/lib -lmiblab -L$(top_srcdir)/liblr -llr
test_classifier_DEPENDENCIES = $(top_srcdir)/lib/libmiblab.a $(top_srcdir)/liblr/liblr.a

test_classifier2_SOURCES = test_classifier2.cpp
test_classifier2_CFLAGS = -g -O2
test_classifier2_CPPFLAGS = -I$(top_srcdir)/include -I$(BOOST)
test_classifier2_LDFLAGS = -L$(top_srcdir)/lib -lmiblab -L$(top_srcdir)/liblr -llr
test_classifier2_DEPENDENCIES = $(top_srcdir)/lib/libmiblab.a $(top_srcdir)/liblr/liblr.a

test_mrmr_SOURCES = test_mrmr.cpp
test_mrmr_CFLAGS = -g -O2
test_mrmr_CPPFLAGS = -I$(top_srcdir)/include -I$(BOOST)
test_mrmr_LDFLAGS = -L$(top_srcdir)/lib -lmiblab
test_mrmr_DEPENDENCIES = $(top_srcdir)/lib/libmiblab.a

test_ranksum_SOURCES = test_ranksum.cpp
test_ranksum_CFLAGS = -g -O2
test_ranksum_CPPFLAGS = -I$(top_srcdir)/include -I$(BOOST)
test_ranksum_LDFLAGS = -L$(top_srcdir)/lib -lmiblab
test_ranksum_DEPENDENCIES = $(top_srcdir)/lib/libmiblab.a

data_combination_SOURCES = data_combination.cpp
data_combination_CFLAGS = -g -O2
data_combination_CPPFLAGS = -I$(top_srcdir)/include -I$(BOOST) -I$(OPENCV)/include
data_combination_LDFLAGS = -L$(top_srcdir)/lib -lmiblab -L$(top_srcdir)/liblr -llr -L$(OPENCV)/lib -lml -lcxcore -lcvaux -lopencv_lapack -lzlib -lrt -lgfortran -L$(top_srcdir)/libsvm-2.8 -lsvm
data_combination_DEPENDENCIES = $(top_srcdir)/lib/libmiblab.a $(top_srcdir)/liblr/liblr.a $(top_srcdir)/libsvm-2.8/libsvm.a

data_combination_rankprod_SOURCES = data_combination_rankprod.cpp
data_combination_rankprod_CFLAGS = -g -O2
data_combination_rankprod_CPPFLAGS = -I$(top_srcdir)/include -I$(BOOST) -I$(OPENCV)/include
data_combination_rankprod_LDFLAGS = -L$(top_srcdir)/lib -lmiblab -L$(top_srcdir)/liblr -llr -L$(OPENCV)/lib -lml -lcxcore -lcvaux -lopencv_lapack -lzlib -lrt -lgfortran -L$(top_srcdir)/libsvm-2.8 -lsvm
data_combination_rankprod_DEPENDENCIES = $(top_srcdir)/lib/libmiblab.a $(top_srcdir)/liblr/liblr.a $(top_srcdir)/libsvm-2.8/libsvm.a

data_combination_choi_SOURCES = data_combination_choi.cpp
data_combination_choi_CFLAGS = -g -O2
data_combination_choi_CPPFLAGS = -I$(top_srcdir)/include -I$(BOOST) -I$(OPENCV)/include
data_combination_choi_LDFLAGS = -L$(top_srcdir)/lib -lmiblab -L$(top_srcdir)/liblr -llr -L$(OPENCV)/lib -lml -lcxcore -lcvaux -lopencv_lapack -lzlib -lrt -lgfortran -L$(top_srcdir)/libsvm-2.8 -lsvm
data_combination_choi_DEPENDENCIES = $(top_srcdir)/lib/libmiblab.a $(top_srcdir)/liblr/liblr.a $(top_srcdir)/libsvm-2.8/libsvm.a

data_combination_wang_SOURCES = data_combination_wang.cpp
data_combination_wang_CFLAGS = -g -O2
data_combination_wang_CPPFLAGS = -I$(top_srcdir)/include -I$(BOOST) -I$(OPENCV)/include
data_combination_wang_LDFLAGS = -L$(top_srcdir)/lib -lmiblab -L$(top_srcdir)/liblr -llr -L$(OPENCV)/lib -lml -lcxcore -lcvaux -lopencv_lapack -lzlib -lrt -lgfortran -L$(top_srcdir)/libsvm-2.8 -lsvm
data_combination_wang_DEPENDENCIES = $(top_srcdir)/lib/libmiblab.a $(top_srcdir)/liblr/liblr.a $(top_srcdir)/libsvm-2.8/libsvm.a

data_combination_mdeds_SOURCES = data_combination_mdeds.cpp
data_combination_mdeds_CFLAGS = -g -O2
data_combination_mdeds_CPPFLAGS = -I$(top_srcdir)/include -I$(BOOST) -I$(OPENCV)/include
data_combination_mdeds_LDFLAGS = -L$(top_srcdir)/lib -lmiblab -L$(top_srcdir)/liblr -llr -L$(OPENCV)/lib -lml -lcxcore -lcvaux -lopencv_lapack -lzlib -lrt -lgfortran -L$(top_srcdir)/libsvm-2.8 -lsvm
data_combination_mdeds_DEPENDENCIES = $(top_srcdir)/lib/libmiblab.a $(top_srcdir)/liblr/liblr.a $(top_srcdir)/libsvm-2.8/libsvm.a

data_combination_mdeds2_SOURCES = data_combination_mdeds2.cpp
data_combination_mdeds2_CFLAGS = -g -O2
data_combination_mdeds2_CPPFLAGS = -I$(top_srcdir)/include -I$(BOOST) -I$(OPENCV)/include
data_combination_mdeds2_LDFLAGS = -L$(top_srcdir)/lib -lmiblab -L$(top_srcdir)/liblr -llr -L$(OPENCV)/lib -lml -lcxcore -lcvaux -lopencv_lapack -lzlib -lrt -lgfortran -L$(top_srcdir)/libsvm-2.8 -lsvm
data_combination_mdeds2_DEPENDENCIES = $(top_srcdir)/lib/libmiblab.a $(top_srcdir)/liblr/liblr.a $(top_srcdir)/libsvm-2.8/libsvm.a

data_combination_one_SOURCES = data_combination_one.cpp
data_combination_one_CFLAGS = -g -O2
data_combination_one_CPPFLAGS = -I$(top_srcdir)/include -I$(BOOST) -I$(OPENCV)/include
data_combination_one_LDFLAGS = -L$(top_srcdir)/lib -lmiblab -L$(top_srcdir)/liblr -llr -L$(OPENCV)/lib -lml -lcxcore -lcvaux -lopencv_lapack -lzlib -lrt -lgfortran -L$(top_srcdir)/libsvm-2.8 -lsvm
data_combination_one_DEPENDENCIES = $(top_srcdir)/lib/libmiblab.a $(top_srcdir)/liblr/liblr.a $(top_srcdir)/libsvm-2.8/libsvm.a

extract_rows_SOURCES = extract_rows.cpp
extract_rows_CFLAGS = -g -O2
extract_rows_CPPFLAGS = -I$(top_srcdir)/include
extract_rows_LDFLAGS = -L$(top_srcdir)/lib -lmiblab
extract_rows_DEPENDENCIES = $(top_srcdir)/lib/libmiblab.a
