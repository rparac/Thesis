??3
?E?E
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	??
?
	ApplyAdam
var"T?	
m"T?	
v"T?
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T?" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
?
ApplyRMSProp
var"T?

ms"T?
mom"T?
lr"T
rho"T
momentum"T
epsilon"T	
grad"T
out"T?" 
Ttype:
2	"
use_lockingbool( 
?
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
s
	AssignSub
ref"T?

value"T

output_ref"T?" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Digamma
x"T
y"T"
Ttype:
2
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
B
Equal
x"T
y"T
z
"
Ttype:
2	
?
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
?
FusedBatchNorm
x"T

scale"T
offset"T	
mean"T
variance"T
y"T

batch_mean"T
batch_variance"T
reserve_space_1"T
reserve_space_2"T"
Ttype:
2"
epsilonfloat%??8"-
data_formatstringNHWC:
NHWCNCHW"
is_trainingbool(
?
FusedBatchNormGrad

y_backprop"T
x"T

scale"T
reserve_space_1"T
reserve_space_2"T

x_backprop"T
scale_backprop"T
offset_backprop"T
reserve_space_3"T
reserve_space_4"T"
Ttype:
2"
epsilonfloat%??8"-
data_formatstringNHWC:
NHWCNCHW"
is_trainingbool(
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
2
L2Loss
t"T
output"T"
Ttype:
2
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
n
LeakyReluGrad
	gradients"T
features"T
	backprops"T"
alphafloat%??L>"
Ttype0:
2
-
Lgamma
x"T
y"T"
Ttype:
2
,
Log
x"T
y"T"
Ttype:

2
.
Log1p
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
	Polygamma
a"T
x"T
z"T"
Ttype:
2
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
s
	ScatterNd
indices"Tindices
updates"T
shape"Tindices
output"T"	
Ttype"
Tindicestype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
/
Sign
x"T
y"T"
Ttype:

2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
@
Softplus
features"T
activations"T"
Ttype:
2
1
Square
x"T
y"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
?
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.14.02v1.14.0-0-g87989f69597??1
f
XPlaceholder*
dtype0*
shape:??????????*(
_output_shapes
:??????????
d
YPlaceholder*
dtype0*'
_output_shapes
:?????????*
shape:?????????
P
PlaceholderPlaceholder*
shape:*
_output_shapes
:*
dtype0
n
encoder/Reshape/shapeConst*
_output_shapes
:*%
valueB"????         *
dtype0
|
encoder/ReshapeReshapeXencoder/Reshape/shape*
T0*/
_output_shapes
:?????????*
Tshape0
?
6encoder/conv2d/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*(
_class
loc:@encoder/conv2d/kernel*%
valueB"            
?
4encoder/conv2d/kernel/Initializer/random_uniform/minConst*
valueB
 *??ڽ*
_output_shapes
: *(
_class
loc:@encoder/conv2d/kernel*
dtype0
?
4encoder/conv2d/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@encoder/conv2d/kernel*
_output_shapes
: *
valueB
 *???=*
dtype0
?
>encoder/conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform6encoder/conv2d/kernel/Initializer/random_uniform/shape*
seed2 *(
_class
loc:@encoder/conv2d/kernel*&
_output_shapes
:*
dtype0*
T0*

seed 
?
4encoder/conv2d/kernel/Initializer/random_uniform/subSub4encoder/conv2d/kernel/Initializer/random_uniform/max4encoder/conv2d/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*(
_class
loc:@encoder/conv2d/kernel
?
4encoder/conv2d/kernel/Initializer/random_uniform/mulMul>encoder/conv2d/kernel/Initializer/random_uniform/RandomUniform4encoder/conv2d/kernel/Initializer/random_uniform/sub*&
_output_shapes
:*
T0*(
_class
loc:@encoder/conv2d/kernel
?
0encoder/conv2d/kernel/Initializer/random_uniformAdd4encoder/conv2d/kernel/Initializer/random_uniform/mul4encoder/conv2d/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@encoder/conv2d/kernel*&
_output_shapes
:
?
encoder/conv2d/kernel
VariableV2*
shared_name *(
_class
loc:@encoder/conv2d/kernel*
dtype0*&
_output_shapes
:*
	container *
shape:
?
encoder/conv2d/kernel/AssignAssignencoder/conv2d/kernel0encoder/conv2d/kernel/Initializer/random_uniform*
use_locking(*(
_class
loc:@encoder/conv2d/kernel*
validate_shape(*&
_output_shapes
:*
T0
?
encoder/conv2d/kernel/readIdentityencoder/conv2d/kernel*(
_class
loc:@encoder/conv2d/kernel*&
_output_shapes
:*
T0
?
%encoder/conv2d/bias/Initializer/zerosConst*&
_class
loc:@encoder/conv2d/bias*
valueB*    *
dtype0*
_output_shapes
:
?
encoder/conv2d/bias
VariableV2*
_output_shapes
:*
shared_name *
	container *
shape:*
dtype0*&
_class
loc:@encoder/conv2d/bias
?
encoder/conv2d/bias/AssignAssignencoder/conv2d/bias%encoder/conv2d/bias/Initializer/zeros*
use_locking(*
_output_shapes
:*
validate_shape(*&
_class
loc:@encoder/conv2d/bias*
T0
?
encoder/conv2d/bias/readIdentityencoder/conv2d/bias*&
_class
loc:@encoder/conv2d/bias*
_output_shapes
:*
T0
m
encoder/conv2d/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
?
encoder/conv2d/Conv2DConv2Dencoder/Reshapeencoder/conv2d/kernel/read*
	dilations
*/
_output_shapes
:?????????*
use_cudnn_on_gpu(*
paddingVALID*
T0*
explicit_paddings
 *
data_formatNHWC*
strides

?
encoder/conv2d/BiasAddBiasAddencoder/conv2d/Conv2Dencoder/conv2d/bias/read*
T0*/
_output_shapes
:?????????*
data_formatNHWC
m
encoder/conv2d/ReluReluencoder/conv2d/BiasAdd*
T0*/
_output_shapes
:?????????
?
encoder/max_pooling2d/MaxPoolMaxPoolencoder/conv2d/Relu*
T0*
paddingVALID*/
_output_shapes
:?????????*
data_formatNHWC*
ksize
*
strides

?
8encoder/conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"         2   *
dtype0*
_output_shapes
:**
_class 
loc:@encoder/conv2d_1/kernel
?
6encoder/conv2d_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: **
_class 
loc:@encoder/conv2d_1/kernel*
valueB
 *S?o?*
dtype0
?
6encoder/conv2d_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *S?o=**
_class 
loc:@encoder/conv2d_1/kernel
?
@encoder/conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform8encoder/conv2d_1/kernel/Initializer/random_uniform/shape*
seed2 *

seed **
_class 
loc:@encoder/conv2d_1/kernel*
dtype0*&
_output_shapes
:2*
T0
?
6encoder/conv2d_1/kernel/Initializer/random_uniform/subSub6encoder/conv2d_1/kernel/Initializer/random_uniform/max6encoder/conv2d_1/kernel/Initializer/random_uniform/min*
_output_shapes
: **
_class 
loc:@encoder/conv2d_1/kernel*
T0
?
6encoder/conv2d_1/kernel/Initializer/random_uniform/mulMul@encoder/conv2d_1/kernel/Initializer/random_uniform/RandomUniform6encoder/conv2d_1/kernel/Initializer/random_uniform/sub*&
_output_shapes
:2**
_class 
loc:@encoder/conv2d_1/kernel*
T0
?
2encoder/conv2d_1/kernel/Initializer/random_uniformAdd6encoder/conv2d_1/kernel/Initializer/random_uniform/mul6encoder/conv2d_1/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@encoder/conv2d_1/kernel*&
_output_shapes
:2
?
encoder/conv2d_1/kernel
VariableV2*
	container *
shared_name *
dtype0*
shape:2**
_class 
loc:@encoder/conv2d_1/kernel*&
_output_shapes
:2
?
encoder/conv2d_1/kernel/AssignAssignencoder/conv2d_1/kernel2encoder/conv2d_1/kernel/Initializer/random_uniform**
_class 
loc:@encoder/conv2d_1/kernel*
use_locking(*
T0*
validate_shape(*&
_output_shapes
:2
?
encoder/conv2d_1/kernel/readIdentityencoder/conv2d_1/kernel*&
_output_shapes
:2*
T0**
_class 
loc:@encoder/conv2d_1/kernel
?
'encoder/conv2d_1/bias/Initializer/zerosConst*
valueB2*    *(
_class
loc:@encoder/conv2d_1/bias*
dtype0*
_output_shapes
:2
?
encoder/conv2d_1/bias
VariableV2*
_output_shapes
:2*
dtype0*(
_class
loc:@encoder/conv2d_1/bias*
shape:2*
	container *
shared_name 
?
encoder/conv2d_1/bias/AssignAssignencoder/conv2d_1/bias'encoder/conv2d_1/bias/Initializer/zeros*(
_class
loc:@encoder/conv2d_1/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:2
?
encoder/conv2d_1/bias/readIdentityencoder/conv2d_1/bias*(
_class
loc:@encoder/conv2d_1/bias*
_output_shapes
:2*
T0
o
encoder/conv2d_1/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
encoder/conv2d_1/Conv2DConv2Dencoder/max_pooling2d/MaxPoolencoder/conv2d_1/kernel/read*
T0*
explicit_paddings
 *
	dilations
*
strides
*
paddingVALID*
data_formatNHWC*
use_cudnn_on_gpu(*/
_output_shapes
:?????????2
?
encoder/conv2d_1/BiasAddBiasAddencoder/conv2d_1/Conv2Dencoder/conv2d_1/bias/read*/
_output_shapes
:?????????2*
data_formatNHWC*
T0
q
encoder/conv2d_1/ReluReluencoder/conv2d_1/BiasAdd*/
_output_shapes
:?????????2*
T0
?
encoder/max_pooling2d_1/MaxPoolMaxPoolencoder/conv2d_1/Relu*
strides
*
T0*
data_formatNHWC*
paddingVALID*
ksize
*/
_output_shapes
:?????????2
t
encoder/flatten/ShapeShapeencoder/max_pooling2d_1/MaxPool*
out_type0*
T0*
_output_shapes
:
m
#encoder/flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%encoder/flatten/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
o
%encoder/flatten/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
encoder/flatten/strided_sliceStridedSliceencoder/flatten/Shape#encoder/flatten/strided_slice/stack%encoder/flatten/strided_slice/stack_1%encoder/flatten/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: *
end_mask *

begin_mask *
ellipsis_mask *
new_axis_mask 
j
encoder/flatten/Reshape/shape/1Const*
_output_shapes
: *
valueB :
?????????*
dtype0
?
encoder/flatten/Reshape/shapePackencoder/flatten/strided_sliceencoder/flatten/Reshape/shape/1*
_output_shapes
:*

axis *
N*
T0
?
encoder/flatten/ReshapeReshapeencoder/max_pooling2d_1/MaxPoolencoder/flatten/Reshape/shape*
T0*(
_output_shapes
:??????????*
Tshape0
?
5encoder/dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*'
_class
loc:@encoder/dense/kernel*
dtype0*
valueB"   d   
?
3encoder/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *'
_class
loc:@encoder/dense/kernel*
dtype0*
valueB
 *?7??
?
3encoder/dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *'
_class
loc:@encoder/dense/kernel*
valueB
 *?7?=*
dtype0
?
=encoder/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5encoder/dense/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*'
_class
loc:@encoder/dense/kernel*
_output_shapes
:	?d
?
3encoder/dense/kernel/Initializer/random_uniform/subSub3encoder/dense/kernel/Initializer/random_uniform/max3encoder/dense/kernel/Initializer/random_uniform/min*'
_class
loc:@encoder/dense/kernel*
_output_shapes
: *
T0
?
3encoder/dense/kernel/Initializer/random_uniform/mulMul=encoder/dense/kernel/Initializer/random_uniform/RandomUniform3encoder/dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	?d*'
_class
loc:@encoder/dense/kernel*
T0
?
/encoder/dense/kernel/Initializer/random_uniformAdd3encoder/dense/kernel/Initializer/random_uniform/mul3encoder/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	?d*
T0*'
_class
loc:@encoder/dense/kernel
?
encoder/dense/kernel
VariableV2*
dtype0*
	container *'
_class
loc:@encoder/dense/kernel*
shape:	?d*
_output_shapes
:	?d*
shared_name 
?
encoder/dense/kernel/AssignAssignencoder/dense/kernel/encoder/dense/kernel/Initializer/random_uniform*
T0*
_output_shapes
:	?d*
validate_shape(*'
_class
loc:@encoder/dense/kernel*
use_locking(
?
encoder/dense/kernel/readIdentityencoder/dense/kernel*'
_class
loc:@encoder/dense/kernel*
T0*
_output_shapes
:	?d
?
$encoder/dense/bias/Initializer/zerosConst*%
_class
loc:@encoder/dense/bias*
dtype0*
valueBd*    *
_output_shapes
:d
?
encoder/dense/bias
VariableV2*
shared_name *
dtype0*
shape:d*%
_class
loc:@encoder/dense/bias*
_output_shapes
:d*
	container 
?
encoder/dense/bias/AssignAssignencoder/dense/bias$encoder/dense/bias/Initializer/zeros*
T0*
_output_shapes
:d*
validate_shape(*%
_class
loc:@encoder/dense/bias*
use_locking(
?
encoder/dense/bias/readIdentityencoder/dense/bias*
T0*%
_class
loc:@encoder/dense/bias*
_output_shapes
:d
?
encoder/dense/MatMulMatMulencoder/flatten/Reshapeencoder/dense/kernel/read*
transpose_a( *
T0*'
_output_shapes
:?????????d*
transpose_b( 
?
encoder/dense/BiasAddBiasAddencoder/dense/MatMulencoder/dense/bias/read*
T0*'
_output_shapes
:?????????d*
data_formatNHWC
?
7encoder/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"   d   *
_output_shapes
:*)
_class
loc:@encoder/dense_1/kernel*
dtype0
?
5encoder/dense_1/kernel/Initializer/random_uniform/minConst*)
_class
loc:@encoder/dense_1/kernel*
dtype0*
valueB
 *?7??*
_output_shapes
: 
?
5encoder/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *?7?=*
_output_shapes
: *
dtype0*)
_class
loc:@encoder/dense_1/kernel
?
?encoder/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform7encoder/dense_1/kernel/Initializer/random_uniform/shape*
seed2 *)
_class
loc:@encoder/dense_1/kernel*

seed *
_output_shapes
:	?d*
dtype0*
T0
?
5encoder/dense_1/kernel/Initializer/random_uniform/subSub5encoder/dense_1/kernel/Initializer/random_uniform/max5encoder/dense_1/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *)
_class
loc:@encoder/dense_1/kernel
?
5encoder/dense_1/kernel/Initializer/random_uniform/mulMul?encoder/dense_1/kernel/Initializer/random_uniform/RandomUniform5encoder/dense_1/kernel/Initializer/random_uniform/sub*
_output_shapes
:	?d*)
_class
loc:@encoder/dense_1/kernel*
T0
?
1encoder/dense_1/kernel/Initializer/random_uniformAdd5encoder/dense_1/kernel/Initializer/random_uniform/mul5encoder/dense_1/kernel/Initializer/random_uniform/min*)
_class
loc:@encoder/dense_1/kernel*
T0*
_output_shapes
:	?d
?
encoder/dense_1/kernel
VariableV2*
shape:	?d*
	container *
shared_name *)
_class
loc:@encoder/dense_1/kernel*
_output_shapes
:	?d*
dtype0
?
encoder/dense_1/kernel/AssignAssignencoder/dense_1/kernel1encoder/dense_1/kernel/Initializer/random_uniform*
validate_shape(*)
_class
loc:@encoder/dense_1/kernel*
_output_shapes
:	?d*
use_locking(*
T0
?
encoder/dense_1/kernel/readIdentityencoder/dense_1/kernel*)
_class
loc:@encoder/dense_1/kernel*
_output_shapes
:	?d*
T0
?
&encoder/dense_1/bias/Initializer/zerosConst*
_output_shapes
:d*'
_class
loc:@encoder/dense_1/bias*
dtype0*
valueBd*    
?
encoder/dense_1/bias
VariableV2*
_output_shapes
:d*
	container *
dtype0*'
_class
loc:@encoder/dense_1/bias*
shape:d*
shared_name 
?
encoder/dense_1/bias/AssignAssignencoder/dense_1/bias&encoder/dense_1/bias/Initializer/zeros*
validate_shape(*'
_class
loc:@encoder/dense_1/bias*
T0*
_output_shapes
:d*
use_locking(
?
encoder/dense_1/bias/readIdentityencoder/dense_1/bias*
_output_shapes
:d*
T0*'
_class
loc:@encoder/dense_1/bias
?
encoder/dense_1/MatMulMatMulencoder/flatten/Reshapeencoder/dense_1/kernel/read*'
_output_shapes
:?????????d*
T0*
transpose_a( *
transpose_b( 
?
encoder/dense_1/BiasAddBiasAddencoder/dense_1/MatMulencoder/dense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:?????????d
o
encoder/dense_1/SoftplusSoftplusencoder/dense_1/BiasAdd*'
_output_shapes
:?????????d*
T0
?
hencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/range_dimension_tensor/ConstConst*
dtype0*
value	B :d*
_output_shapes
: 
?
Dencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
valueB:d*
_output_shapes
:*
dtype0
?
?encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShapeencoder/dense_1/Softplus*
out_type0*
T0*
_output_shapes
:
?
?encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
valueB:
?????????*
dtype0
?
?encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
valueB: *
dtype0
?
?encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
?encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSlice?encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape?encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack?encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1?encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2*
shrink_axis_mask*
T0*

begin_mask *
_output_shapes
: *
ellipsis_mask *
Index0*
new_axis_mask *
end_mask 
?
?encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Pack?encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice*
T0*
N*
_output_shapes
:*

axis 
?
?encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
?
?encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2?encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape?encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1?encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis*
_output_shapes
:*

Tidx0*
N*
T0
?
rencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
tencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????
?
tencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
lencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSlice?encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatrencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stacktencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1tencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2*
ellipsis_mask *
end_mask *
shrink_axis_mask *
_output_shapes
:*
T0*
Index0*
new_axis_mask *

begin_mask
?
>encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShapeencoder/dense/BiasAdd*
_output_shapes
:*
out_type0*
T0
?
Lencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Nencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
valueB:
?????????*
dtype0
?
Nencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
Fencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlice>encoder/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeLencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackNencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Nencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2*
ellipsis_mask *
shrink_axis_mask *

begin_mask*
Index0*
new_axis_mask *
_output_shapes
:*
end_mask *
T0
?
dencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgsBroadcastArgslencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceFencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice*
_output_shapes
:*
T0
R
encoder/zerosConst*
valueB
 *    *
_output_shapes
: *
dtype0
Q
encoder/onesConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
S
encoder/Normal/IdentityIdentityencoder/zeros*
T0*
_output_shapes
: 
T
encoder/Normal/Identity_1Identityencoder/ones*
T0*
_output_shapes
: 
e
#encoder/MultivariateNormalDiag/zeroConst*
value	B : *
dtype0*
_output_shapes
: 
g
$encoder/MultivariateNormalDiag/emptyConst*
dtype0*
_output_shapes
: *
valueB 
p
.encoder/Normal/is_scalar_batch/is_scalar_batchConst*
_output_shapes
: *
dtype0
*
value	B
 Z
p
.encoder/Normal/is_scalar_event/is_scalar_eventConst*
_output_shapes
: *
dtype0
*
value	B
 Z
r
0encoder/Normal/is_scalar_batch_1/is_scalar_batchConst*
_output_shapes
: *
dtype0
*
value	B
 Z
f
$encoder/MultivariateNormalDiag/ConstConst*
_output_shapes
: *
value	B
 Z *
dtype0

u
*encoder/MultivariateNormalDiag/range/startConst*
valueB :
?????????*
_output_shapes
: *
dtype0
l
*encoder/MultivariateNormalDiag/range/limitConst*
_output_shapes
: *
dtype0*
value	B : 
l
*encoder/MultivariateNormalDiag/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
$encoder/MultivariateNormalDiag/rangeRange*encoder/MultivariateNormalDiag/range/start*encoder/MultivariateNormalDiag/range/limit*encoder/MultivariateNormalDiag/range/delta*
_output_shapes
:*

Tidx0
u
2encoder/MultivariateNormalDiag/sample/sample_shapeConst*
_output_shapes
: *
valueB *
dtype0
x
6encoder/MultivariateNormalDiag/sample/pick_vector/condConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
?
>encoder/MultivariateNormalDiag/sample/pick_vector/false_vectorConst*
dtype0*
valueB:*
_output_shapes
:
z
8encoder/MultivariateNormalDiag/sample/pick_vector_1/condConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
?
?encoder/MultivariateNormalDiag/sample/pick_vector_1/true_vectorConst*
valueB:*
_output_shapes
:*
dtype0
s
1encoder/MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
,encoder/MultivariateNormalDiag/sample/concatConcatV2>encoder/MultivariateNormalDiag/sample/pick_vector/false_vectordencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgsDencoder/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shape$encoder/MultivariateNormalDiag/empty1encoder/MultivariateNormalDiag/sample/concat/axis*
_output_shapes
:*
N*
T0*

Tidx0
e
encoder/Normal/sample/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
encoder/Normal/sample/ProdProd,encoder/MultivariateNormalDiag/sample/concatencoder/Normal/sample/Const*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
p
-encoder/Normal/batch_shape_tensor/batch_shapeConst*
valueB *
dtype0*
_output_shapes
: 
?
%encoder/Normal/sample/concat/values_0Packencoder/Normal/sample/Prod*

axis *
_output_shapes
:*
N*
T0
c
!encoder/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
encoder/Normal/sample/concatConcatV2%encoder/Normal/sample/concat/values_0-encoder/Normal/batch_shape_tensor/batch_shape!encoder/Normal/sample/concat/axis*
_output_shapes
:*
N*
T0*

Tidx0
m
(encoder/Normal/sample/random_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
o
*encoder/Normal/sample/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
8encoder/Normal/sample/random_normal/RandomStandardNormalRandomStandardNormalencoder/Normal/sample/concat*

seed *
seed2 *
T0*#
_output_shapes
:?????????*
dtype0
?
'encoder/Normal/sample/random_normal/mulMul8encoder/Normal/sample/random_normal/RandomStandardNormal*encoder/Normal/sample/random_normal/stddev*#
_output_shapes
:?????????*
T0
?
#encoder/Normal/sample/random_normalAdd'encoder/Normal/sample/random_normal/mul(encoder/Normal/sample/random_normal/mean*
T0*#
_output_shapes
:?????????
?
encoder/Normal/sample/mulMul#encoder/Normal/sample/random_normalencoder/Normal/Identity_1*#
_output_shapes
:?????????*
T0
?
encoder/Normal/sample/addAddencoder/Normal/sample/mulencoder/Normal/Identity*
T0*#
_output_shapes
:?????????
t
encoder/Normal/sample/ShapeShapeencoder/Normal/sample/add*
out_type0*
T0*
_output_shapes
:
s
)encoder/Normal/sample/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
u
+encoder/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
valueB: *
dtype0
u
+encoder/Normal/sample/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
#encoder/Normal/sample/strided_sliceStridedSliceencoder/Normal/sample/Shape)encoder/Normal/sample/strided_slice/stack+encoder/Normal/sample/strided_slice/stack_1+encoder/Normal/sample/strided_slice/stack_2*
end_mask*
new_axis_mask *
shrink_axis_mask *
_output_shapes
: *
ellipsis_mask *
Index0*

begin_mask *
T0
e
#encoder/Normal/sample/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
?
encoder/Normal/sample/concat_1ConcatV2,encoder/MultivariateNormalDiag/sample/concat#encoder/Normal/sample/strided_slice#encoder/Normal/sample/concat_1/axis*
N*
T0*
_output_shapes
:*

Tidx0
?
encoder/Normal/sample/ReshapeReshapeencoder/Normal/sample/addencoder/Normal/sample/concat_1*
T0*
Tshape0*+
_output_shapes
:?????????d
?
+encoder/MultivariateNormalDiag/sample/ShapeShapeencoder/Normal/sample/Reshape*
out_type0*
T0*
_output_shapes
:
?
9encoder/MultivariateNormalDiag/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
;encoder/MultivariateNormalDiag/sample/strided_slice/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
?
;encoder/MultivariateNormalDiag/sample/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
3encoder/MultivariateNormalDiag/sample/strided_sliceStridedSlice+encoder/MultivariateNormalDiag/sample/Shape9encoder/MultivariateNormalDiag/sample/strided_slice/stack;encoder/MultivariateNormalDiag/sample/strided_slice/stack_1;encoder/MultivariateNormalDiag/sample/strided_slice/stack_2*
_output_shapes
:*
shrink_axis_mask *
end_mask*
ellipsis_mask *
new_axis_mask *
T0*
Index0*

begin_mask 
u
3encoder/MultivariateNormalDiag/sample/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
.encoder/MultivariateNormalDiag/sample/concat_1ConcatV22encoder/MultivariateNormalDiag/sample/sample_shape3encoder/MultivariateNormalDiag/sample/strided_slice3encoder/MultivariateNormalDiag/sample/concat_1/axis*
_output_shapes
:*
T0*
N*

Tidx0
?
-encoder/MultivariateNormalDiag/sample/ReshapeReshapeencoder/Normal/sample/Reshape.encoder/MultivariateNormalDiag/sample/concat_1*
T0*
Tshape0*'
_output_shapes
:?????????d
?
bencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mulMulencoder/dense_1/Softplus-encoder/MultivariateNormalDiag/sample/Reshape*'
_output_shapes
:?????????d*
T0
?
Hencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/addAddbencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mulencoder/dense/BiasAdd*'
_output_shapes
:?????????d*
T0
?
	gen/ShapeShapeHencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add*
T0*
out_type0*
_output_shapes
:
a
gen/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
c
gen/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
c
gen/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
?
gen/strided_sliceStridedSlice	gen/Shapegen/strided_slice/stackgen/strided_slice/stack_1gen/strided_slice/stack_2*
_output_shapes
: *
T0*
Index0*
ellipsis_mask *
new_axis_mask *
end_mask *
shrink_axis_mask*

begin_mask 
[
gen/random_normal/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
?
gen/random_normal/shapePackgen/strided_slicegen/random_normal/shape/1*
T0*
N*

axis *
_output_shapes
:
[
gen/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
]
gen/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
&gen/random_normal/RandomStandardNormalRandomStandardNormalgen/random_normal/shape*
seed2 *
dtype0*

seed *
T0*'
_output_shapes
:?????????
?
gen/random_normal/mulMul&gen/random_normal/RandomStandardNormalgen/random_normal/stddev*'
_output_shapes
:?????????*
T0
y
gen/random_normalAddgen/random_normal/mulgen/random_normal/mean*'
_output_shapes
:?????????*
T0
Q
gen/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
?

gen/concatConcatV2gen/random_normalHencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/addgen/concat/axis*
N*
T0*'
_output_shapes
:?????????f*

Tidx0
?
1gen/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"f       *
_output_shapes
:*#
_class
loc:@gen/dense/kernel
?
/gen/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *??X?*
dtype0*
_output_shapes
: *#
_class
loc:@gen/dense/kernel
?
/gen/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*#
_class
loc:@gen/dense/kernel*
_output_shapes
: *
valueB
 *??X>
?
9gen/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform1gen/dense/kernel/Initializer/random_uniform/shape*#
_class
loc:@gen/dense/kernel*

seed *
T0*
seed2 *
_output_shapes

:f *
dtype0
?
/gen/dense/kernel/Initializer/random_uniform/subSub/gen/dense/kernel/Initializer/random_uniform/max/gen/dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *#
_class
loc:@gen/dense/kernel
?
/gen/dense/kernel/Initializer/random_uniform/mulMul9gen/dense/kernel/Initializer/random_uniform/RandomUniform/gen/dense/kernel/Initializer/random_uniform/sub*#
_class
loc:@gen/dense/kernel*
T0*
_output_shapes

:f 
?
+gen/dense/kernel/Initializer/random_uniformAdd/gen/dense/kernel/Initializer/random_uniform/mul/gen/dense/kernel/Initializer/random_uniform/min*
_output_shapes

:f *#
_class
loc:@gen/dense/kernel*
T0
?
gen/dense/kernel
VariableV2*
	container *
dtype0*
shared_name *#
_class
loc:@gen/dense/kernel*
_output_shapes

:f *
shape
:f 
?
gen/dense/kernel/AssignAssigngen/dense/kernel+gen/dense/kernel/Initializer/random_uniform*#
_class
loc:@gen/dense/kernel*
validate_shape(*
T0*
_output_shapes

:f *
use_locking(
?
gen/dense/kernel/readIdentitygen/dense/kernel*
_output_shapes

:f *
T0*#
_class
loc:@gen/dense/kernel
?
 gen/dense/bias/Initializer/zerosConst*!
_class
loc:@gen/dense/bias*
dtype0*
_output_shapes
: *
valueB *    
?
gen/dense/bias
VariableV2*
_output_shapes
: *
shared_name *!
_class
loc:@gen/dense/bias*
dtype0*
	container *
shape: 
?
gen/dense/bias/AssignAssigngen/dense/bias gen/dense/bias/Initializer/zeros*!
_class
loc:@gen/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
w
gen/dense/bias/readIdentitygen/dense/bias*
T0*
_output_shapes
: *!
_class
loc:@gen/dense/bias
?
gen/dense/MatMulMatMul
gen/concatgen/dense/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:????????? 
?
gen/dense/BiasAddBiasAddgen/dense/MatMulgen/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:????????? 
u
gen/dense/LeakyRelu	LeakyRelugen/dense/BiasAdd*
T0*'
_output_shapes
:????????? *
alpha%??L>
?
3gen/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"        *%
_class
loc:@gen/dense_1/kernel*
_output_shapes
:
?
1gen/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *%
_class
loc:@gen/dense_1/kernel*
valueB
 *qĜ?*
dtype0
?
1gen/dense_1/kernel/Initializer/random_uniform/maxConst*%
_class
loc:@gen/dense_1/kernel*
dtype0*
valueB
 *qĜ>*
_output_shapes
: 
?
;gen/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform3gen/dense_1/kernel/Initializer/random_uniform/shape*

seed *%
_class
loc:@gen/dense_1/kernel*
_output_shapes

:  *
seed2 *
dtype0*
T0
?
1gen/dense_1/kernel/Initializer/random_uniform/subSub1gen/dense_1/kernel/Initializer/random_uniform/max1gen/dense_1/kernel/Initializer/random_uniform/min*
T0*%
_class
loc:@gen/dense_1/kernel*
_output_shapes
: 
?
1gen/dense_1/kernel/Initializer/random_uniform/mulMul;gen/dense_1/kernel/Initializer/random_uniform/RandomUniform1gen/dense_1/kernel/Initializer/random_uniform/sub*
_output_shapes

:  *%
_class
loc:@gen/dense_1/kernel*
T0
?
-gen/dense_1/kernel/Initializer/random_uniformAdd1gen/dense_1/kernel/Initializer/random_uniform/mul1gen/dense_1/kernel/Initializer/random_uniform/min*
T0*%
_class
loc:@gen/dense_1/kernel*
_output_shapes

:  
?
gen/dense_1/kernel
VariableV2*
dtype0*
shared_name *%
_class
loc:@gen/dense_1/kernel*
_output_shapes

:  *
shape
:  *
	container 
?
gen/dense_1/kernel/AssignAssigngen/dense_1/kernel-gen/dense_1/kernel/Initializer/random_uniform*
use_locking(*
validate_shape(*%
_class
loc:@gen/dense_1/kernel*
T0*
_output_shapes

:  
?
gen/dense_1/kernel/readIdentitygen/dense_1/kernel*
_output_shapes

:  *
T0*%
_class
loc:@gen/dense_1/kernel
?
"gen/dense_1/bias/Initializer/zerosConst*
_output_shapes
: *
valueB *    *
dtype0*#
_class
loc:@gen/dense_1/bias
?
gen/dense_1/bias
VariableV2*
	container *
shared_name *
dtype0*#
_class
loc:@gen/dense_1/bias*
_output_shapes
: *
shape: 
?
gen/dense_1/bias/AssignAssigngen/dense_1/bias"gen/dense_1/bias/Initializer/zeros*
validate_shape(*
_output_shapes
: *#
_class
loc:@gen/dense_1/bias*
T0*
use_locking(
}
gen/dense_1/bias/readIdentitygen/dense_1/bias*
T0*#
_class
loc:@gen/dense_1/bias*
_output_shapes
: 
?
gen/dense_1/MatMulMatMulgen/dense/LeakyRelugen/dense_1/kernel/read*'
_output_shapes
:????????? *
transpose_a( *
T0*
transpose_b( 
?
gen/dense_1/BiasAddBiasAddgen/dense_1/MatMulgen/dense_1/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:????????? 
y
gen/dense_1/LeakyRelu	LeakyRelugen/dense_1/BiasAdd*
alpha%??L>*
T0*'
_output_shapes
:????????? 
?
3gen/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"        *
dtype0*%
_class
loc:@gen/dense_2/kernel*
_output_shapes
:
?
1gen/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *qĜ?*
_output_shapes
: *%
_class
loc:@gen/dense_2/kernel*
dtype0
?
1gen/dense_2/kernel/Initializer/random_uniform/maxConst*
dtype0*%
_class
loc:@gen/dense_2/kernel*
_output_shapes
: *
valueB
 *qĜ>
?
;gen/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform3gen/dense_2/kernel/Initializer/random_uniform/shape*
_output_shapes

:  *

seed *%
_class
loc:@gen/dense_2/kernel*
T0*
dtype0*
seed2 
?
1gen/dense_2/kernel/Initializer/random_uniform/subSub1gen/dense_2/kernel/Initializer/random_uniform/max1gen/dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *%
_class
loc:@gen/dense_2/kernel
?
1gen/dense_2/kernel/Initializer/random_uniform/mulMul;gen/dense_2/kernel/Initializer/random_uniform/RandomUniform1gen/dense_2/kernel/Initializer/random_uniform/sub*
T0*%
_class
loc:@gen/dense_2/kernel*
_output_shapes

:  
?
-gen/dense_2/kernel/Initializer/random_uniformAdd1gen/dense_2/kernel/Initializer/random_uniform/mul1gen/dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes

:  *%
_class
loc:@gen/dense_2/kernel
?
gen/dense_2/kernel
VariableV2*
_output_shapes

:  *
shape
:  *
dtype0*
shared_name *
	container *%
_class
loc:@gen/dense_2/kernel
?
gen/dense_2/kernel/AssignAssigngen/dense_2/kernel-gen/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*%
_class
loc:@gen/dense_2/kernel*
_output_shapes

:  *
T0
?
gen/dense_2/kernel/readIdentitygen/dense_2/kernel*
_output_shapes

:  *%
_class
loc:@gen/dense_2/kernel*
T0
?
"gen/dense_2/bias/Initializer/zerosConst*
valueB *    *#
_class
loc:@gen/dense_2/bias*
dtype0*
_output_shapes
: 
?
gen/dense_2/bias
VariableV2*#
_class
loc:@gen/dense_2/bias*
	container *
shape: *
dtype0*
shared_name *
_output_shapes
: 
?
gen/dense_2/bias/AssignAssigngen/dense_2/bias"gen/dense_2/bias/Initializer/zeros*#
_class
loc:@gen/dense_2/bias*
T0*
use_locking(*
_output_shapes
: *
validate_shape(
}
gen/dense_2/bias/readIdentitygen/dense_2/bias*
_output_shapes
: *
T0*#
_class
loc:@gen/dense_2/bias
?
gen/dense_2/MatMulMatMulgen/dense_1/LeakyRelugen/dense_2/kernel/read*'
_output_shapes
:????????? *
transpose_b( *
T0*
transpose_a( 
?
gen/dense_2/BiasAddBiasAddgen/dense_2/MatMulgen/dense_2/bias/read*'
_output_shapes
:????????? *
T0*
data_formatNHWC
y
gen/dense_2/LeakyRelu	LeakyRelugen/dense_2/BiasAdd*
alpha%??L>*'
_output_shapes
:????????? *
T0
?
3gen/dense_3/kernel/Initializer/random_uniform/shapeConst*
valueB"    d   *
dtype0*%
_class
loc:@gen/dense_3/kernel*
_output_shapes
:
?
1gen/dense_3/kernel/Initializer/random_uniform/minConst*%
_class
loc:@gen/dense_3/kernel*
dtype0*
valueB
 *JQZ?*
_output_shapes
: 
?
1gen/dense_3/kernel/Initializer/random_uniform/maxConst*%
_class
loc:@gen/dense_3/kernel*
valueB
 *JQZ>*
dtype0*
_output_shapes
: 
?
;gen/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform3gen/dense_3/kernel/Initializer/random_uniform/shape*
dtype0*
T0*

seed *
seed2 *%
_class
loc:@gen/dense_3/kernel*
_output_shapes

: d
?
1gen/dense_3/kernel/Initializer/random_uniform/subSub1gen/dense_3/kernel/Initializer/random_uniform/max1gen/dense_3/kernel/Initializer/random_uniform/min*%
_class
loc:@gen/dense_3/kernel*
T0*
_output_shapes
: 
?
1gen/dense_3/kernel/Initializer/random_uniform/mulMul;gen/dense_3/kernel/Initializer/random_uniform/RandomUniform1gen/dense_3/kernel/Initializer/random_uniform/sub*%
_class
loc:@gen/dense_3/kernel*
T0*
_output_shapes

: d
?
-gen/dense_3/kernel/Initializer/random_uniformAdd1gen/dense_3/kernel/Initializer/random_uniform/mul1gen/dense_3/kernel/Initializer/random_uniform/min*
T0*%
_class
loc:@gen/dense_3/kernel*
_output_shapes

: d
?
gen/dense_3/kernel
VariableV2*
dtype0*%
_class
loc:@gen/dense_3/kernel*
_output_shapes

: d*
shape
: d*
shared_name *
	container 
?
gen/dense_3/kernel/AssignAssigngen/dense_3/kernel-gen/dense_3/kernel/Initializer/random_uniform*%
_class
loc:@gen/dense_3/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes

: d
?
gen/dense_3/kernel/readIdentitygen/dense_3/kernel*
T0*%
_class
loc:@gen/dense_3/kernel*
_output_shapes

: d
?
"gen/dense_3/bias/Initializer/zerosConst*
dtype0*#
_class
loc:@gen/dense_3/bias*
valueBd*    *
_output_shapes
:d
?
gen/dense_3/bias
VariableV2*
	container *
shape:d*#
_class
loc:@gen/dense_3/bias*
shared_name *
_output_shapes
:d*
dtype0
?
gen/dense_3/bias/AssignAssigngen/dense_3/bias"gen/dense_3/bias/Initializer/zeros*
_output_shapes
:d*
T0*#
_class
loc:@gen/dense_3/bias*
use_locking(*
validate_shape(
}
gen/dense_3/bias/readIdentitygen/dense_3/bias*#
_class
loc:@gen/dense_3/bias*
_output_shapes
:d*
T0
?
gen/dense_3/MatMulMatMulgen/dense_2/LeakyRelugen/dense_3/kernel/read*'
_output_shapes
:?????????d*
T0*
transpose_a( *
transpose_b( 
?
gen/dense_3/BiasAddBiasAddgen/dense_3/MatMulgen/dense_3/bias/read*'
_output_shapes
:?????????d*
data_formatNHWC*
T0
g
gen/dense_3/SoftplusSoftplusgen/dense_3/BiasAdd*
T0*'
_output_shapes
:?????????d
J
add/yConst*
valueB
 *o?:*
_output_shapes
: *
dtype0
Y
addAddgen/dense_3/Softplusadd/y*'
_output_shapes
:?????????d*
T0
?
`MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/range_dimension_tensor/ConstConst*
dtype0*
value	B :d*
_output_shapes
: 
?
<MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
valueB:d*
dtype0*
_output_shapes
:
?
|MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShapeadd*
out_type0*
_output_shapes
:*
T0
?
?MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?
?MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
valueB: *
dtype0
?
?MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
?MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSlice|MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape?MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack?MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1?MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
T0*
Index0*
end_mask *
_output_shapes
: *
new_axis_mask 
?
?MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Pack?MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice*
T0*
_output_shapes
:*

axis *
N
?
?MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
}MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2|MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape?MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1?MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis*
_output_shapes
:*
T0*
N*

Tidx0
?
jMultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
?
lMultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
valueB:
?????????*
dtype0*
_output_shapes
:
?
lMultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
dMultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSlice}MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatjMultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stacklMultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1lMultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2*
end_mask *
_output_shapes
:*

begin_mask*
shrink_axis_mask *
Index0*
new_axis_mask *
ellipsis_mask *
T0
?
6MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShapeHencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add*
_output_shapes
:*
out_type0*
T0
?
DMultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
FMultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
valueB:
?????????*
dtype0*
_output_shapes
:
?
FMultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
>MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlice6MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeDMultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackFMultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1FMultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2*
_output_shapes
:*
ellipsis_mask *
shrink_axis_mask *
end_mask *

begin_mask*
T0*
new_axis_mask *
Index0
?
\MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgsBroadcastArgsdMultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice>MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice*
T0*
_output_shapes
:
J
zerosConst*
dtype0*
_output_shapes
: *
valueB
 *    
I
onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
C
Normal/IdentityIdentityzeros*
T0*
_output_shapes
: 
D
Normal/Identity_1Identityones*
T0*
_output_shapes
: 
]
MultivariateNormalDiag/zeroConst*
dtype0*
_output_shapes
: *
value	B : 
_
MultivariateNormalDiag/emptyConst*
dtype0*
valueB *
_output_shapes
: 
h
&Normal/is_scalar_batch/is_scalar_batchConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
h
&Normal/is_scalar_event/is_scalar_eventConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
j
(Normal/is_scalar_batch_1/is_scalar_batchConst*
value	B
 Z*
_output_shapes
: *
dtype0

^
MultivariateNormalDiag/ConstConst*
value	B
 Z *
_output_shapes
: *
dtype0

m
"MultivariateNormalDiag/range/startConst*
dtype0*
_output_shapes
: *
valueB :
?????????
d
"MultivariateNormalDiag/range/limitConst*
dtype0*
value	B : *
_output_shapes
: 
d
"MultivariateNormalDiag/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
?
MultivariateNormalDiag/rangeRange"MultivariateNormalDiag/range/start"MultivariateNormalDiag/range/limit"MultivariateNormalDiag/range/delta*

Tidx0*
_output_shapes
:
m
*MultivariateNormalDiag/sample/sample_shapeConst*
dtype0*
valueB *
_output_shapes
: 
p
.MultivariateNormalDiag/sample/pick_vector/condConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
?
6MultivariateNormalDiag/sample/pick_vector/false_vectorConst*
_output_shapes
:*
dtype0*
valueB:
r
0MultivariateNormalDiag/sample/pick_vector_1/condConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
?
7MultivariateNormalDiag/sample/pick_vector_1/true_vectorConst*
_output_shapes
:*
dtype0*
valueB:
k
)MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
?
$MultivariateNormalDiag/sample/concatConcatV26MultivariateNormalDiag/sample/pick_vector/false_vector\MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgs<MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeMultivariateNormalDiag/empty)MultivariateNormalDiag/sample/concat/axis*

Tidx0*
_output_shapes
:*
N*
T0
]
Normal/sample/ConstConst*
valueB: *
_output_shapes
:*
dtype0
?
Normal/sample/ProdProd$MultivariateNormalDiag/sample/concatNormal/sample/Const*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
h
%Normal/batch_shape_tensor/batch_shapeConst*
_output_shapes
: *
valueB *
dtype0
s
Normal/sample/concat/values_0PackNormal/sample/Prod*
N*

axis *
_output_shapes
:*
T0
[
Normal/sample/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Normal/sample/concatConcatV2Normal/sample/concat/values_0%Normal/batch_shape_tensor/batch_shapeNormal/sample/concat/axis*
T0*
N*

Tidx0*
_output_shapes
:
e
 Normal/sample/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
g
"Normal/sample/random_normal/stddevConst*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
0Normal/sample/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat*
seed2 *#
_output_shapes
:?????????*
dtype0*
T0*

seed 
?
Normal/sample/random_normal/mulMul0Normal/sample/random_normal/RandomStandardNormal"Normal/sample/random_normal/stddev*#
_output_shapes
:?????????*
T0
?
Normal/sample/random_normalAddNormal/sample/random_normal/mul Normal/sample/random_normal/mean*#
_output_shapes
:?????????*
T0
v
Normal/sample/mulMulNormal/sample/random_normalNormal/Identity_1*#
_output_shapes
:?????????*
T0
j
Normal/sample/addAddNormal/sample/mulNormal/Identity*#
_output_shapes
:?????????*
T0
d
Normal/sample/ShapeShapeNormal/sample/add*
T0*
_output_shapes
:*
out_type0
k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
valueB: *
dtype0
m
#Normal/sample/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Normal/sample/strided_sliceStridedSliceNormal/sample/Shape!Normal/sample/strided_slice/stack#Normal/sample/strided_slice/stack_1#Normal/sample/strided_slice/stack_2*
ellipsis_mask *
new_axis_mask *
Index0*
end_mask*

begin_mask *
shrink_axis_mask *
T0*
_output_shapes
: 
]
Normal/sample/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
?
Normal/sample/concat_1ConcatV2$MultivariateNormalDiag/sample/concatNormal/sample/strided_sliceNormal/sample/concat_1/axis*
N*
_output_shapes
:*
T0*

Tidx0
?
Normal/sample/ReshapeReshapeNormal/sample/addNormal/sample/concat_1*+
_output_shapes
:?????????d*
Tshape0*
T0
x
#MultivariateNormalDiag/sample/ShapeShapeNormal/sample/Reshape*
out_type0*
_output_shapes
:*
T0
{
1MultivariateNormalDiag/sample/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
}
3MultivariateNormalDiag/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
}
3MultivariateNormalDiag/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
+MultivariateNormalDiag/sample/strided_sliceStridedSlice#MultivariateNormalDiag/sample/Shape1MultivariateNormalDiag/sample/strided_slice/stack3MultivariateNormalDiag/sample/strided_slice/stack_13MultivariateNormalDiag/sample/strided_slice/stack_2*
_output_shapes
:*
new_axis_mask *

begin_mask *
Index0*
end_mask*
ellipsis_mask *
T0*
shrink_axis_mask 
m
+MultivariateNormalDiag/sample/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
?
&MultivariateNormalDiag/sample/concat_1ConcatV2*MultivariateNormalDiag/sample/sample_shape+MultivariateNormalDiag/sample/strided_slice+MultivariateNormalDiag/sample/concat_1/axis*
_output_shapes
:*
T0*

Tidx0*
N
?
%MultivariateNormalDiag/sample/ReshapeReshapeNormal/sample/Reshape&MultivariateNormalDiag/sample/concat_1*
T0*
Tshape0*'
_output_shapes
:?????????d
?
ZMultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mulMuladd%MultivariateNormalDiag/sample/Reshape*
T0*'
_output_shapes
:?????????d
?
@MultivariateNormalDiag/sample/affine_linear_operator/forward/addAddZMultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mulHencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add*
T0*'
_output_shapes
:?????????d
?
1diz/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"d       *
dtype0*
_output_shapes
:*#
_class
loc:@diz/dense/kernel
?
/diz/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *JQZ?*#
_class
loc:@diz/dense/kernel*
_output_shapes
: *
dtype0
?
/diz/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *JQZ>*
_output_shapes
: *#
_class
loc:@diz/dense/kernel*
dtype0
?
9diz/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform1diz/dense/kernel/Initializer/random_uniform/shape*
_output_shapes

:d *#
_class
loc:@diz/dense/kernel*
seed2 *

seed *
T0*
dtype0
?
/diz/dense/kernel/Initializer/random_uniform/subSub/diz/dense/kernel/Initializer/random_uniform/max/diz/dense/kernel/Initializer/random_uniform/min*#
_class
loc:@diz/dense/kernel*
_output_shapes
: *
T0
?
/diz/dense/kernel/Initializer/random_uniform/mulMul9diz/dense/kernel/Initializer/random_uniform/RandomUniform/diz/dense/kernel/Initializer/random_uniform/sub*#
_class
loc:@diz/dense/kernel*
_output_shapes

:d *
T0
?
+diz/dense/kernel/Initializer/random_uniformAdd/diz/dense/kernel/Initializer/random_uniform/mul/diz/dense/kernel/Initializer/random_uniform/min*
_output_shapes

:d *#
_class
loc:@diz/dense/kernel*
T0
?
diz/dense/kernel
VariableV2*
shape
:d *
dtype0*
_output_shapes

:d *#
_class
loc:@diz/dense/kernel*
	container *
shared_name 
?
diz/dense/kernel/AssignAssigndiz/dense/kernel+diz/dense/kernel/Initializer/random_uniform*
T0*
use_locking(*
_output_shapes

:d *#
_class
loc:@diz/dense/kernel*
validate_shape(
?
diz/dense/kernel/readIdentitydiz/dense/kernel*
_output_shapes

:d *#
_class
loc:@diz/dense/kernel*
T0
v
1diz/dense/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o?:*
_output_shapes
: *
dtype0
t
2diz/dense/kernel/Regularizer/l2_regularizer/L2LossL2Lossdiz/dense/kernel/read*
_output_shapes
: *
T0
?
+diz/dense/kernel/Regularizer/l2_regularizerMul1diz/dense/kernel/Regularizer/l2_regularizer/scale2diz/dense/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
 diz/dense/bias/Initializer/zerosConst*
valueB *    *
dtype0*!
_class
loc:@diz/dense/bias*
_output_shapes
: 
?
diz/dense/bias
VariableV2*
dtype0*
	container *
shape: *!
_class
loc:@diz/dense/bias*
_output_shapes
: *
shared_name 
?
diz/dense/bias/AssignAssigndiz/dense/bias diz/dense/bias/Initializer/zeros*
_output_shapes
: *!
_class
loc:@diz/dense/bias*
T0*
validate_shape(*
use_locking(
w
diz/dense/bias/readIdentitydiz/dense/bias*
_output_shapes
: *!
_class
loc:@diz/dense/bias*
T0
t
/diz/dense/bias/Regularizer/l2_regularizer/scaleConst*
valueB
 *o?:*
dtype0*
_output_shapes
: 
p
0diz/dense/bias/Regularizer/l2_regularizer/L2LossL2Lossdiz/dense/bias/read*
_output_shapes
: *
T0
?
)diz/dense/bias/Regularizer/l2_regularizerMul/diz/dense/bias/Regularizer/l2_regularizer/scale0diz/dense/bias/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
diz/dense/MatMulMatMulHencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/adddiz/dense/kernel/read*
transpose_b( *'
_output_shapes
:????????? *
T0*
transpose_a( 
?
diz/dense/BiasAddBiasAdddiz/dense/MatMuldiz/dense/bias/read*
T0*'
_output_shapes
:????????? *
data_formatNHWC
u
diz/dense/LeakyRelu	LeakyReludiz/dense/BiasAdd*'
_output_shapes
:????????? *
T0*
alpha%??L>
?
3diz/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"        *%
_class
loc:@diz/dense_1/kernel*
dtype0
?
1diz/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*%
_class
loc:@diz/dense_1/kernel*
valueB
 *qĜ?*
_output_shapes
: 
?
1diz/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *qĜ>*%
_class
loc:@diz/dense_1/kernel*
_output_shapes
: *
dtype0
?
;diz/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform3diz/dense_1/kernel/Initializer/random_uniform/shape*
seed2 *%
_class
loc:@diz/dense_1/kernel*
T0*
dtype0*

seed *
_output_shapes

:  
?
1diz/dense_1/kernel/Initializer/random_uniform/subSub1diz/dense_1/kernel/Initializer/random_uniform/max1diz/dense_1/kernel/Initializer/random_uniform/min*%
_class
loc:@diz/dense_1/kernel*
_output_shapes
: *
T0
?
1diz/dense_1/kernel/Initializer/random_uniform/mulMul;diz/dense_1/kernel/Initializer/random_uniform/RandomUniform1diz/dense_1/kernel/Initializer/random_uniform/sub*
T0*%
_class
loc:@diz/dense_1/kernel*
_output_shapes

:  
?
-diz/dense_1/kernel/Initializer/random_uniformAdd1diz/dense_1/kernel/Initializer/random_uniform/mul1diz/dense_1/kernel/Initializer/random_uniform/min*%
_class
loc:@diz/dense_1/kernel*
T0*
_output_shapes

:  
?
diz/dense_1/kernel
VariableV2*
dtype0*
shape
:  *
shared_name *
_output_shapes

:  *
	container *%
_class
loc:@diz/dense_1/kernel
?
diz/dense_1/kernel/AssignAssigndiz/dense_1/kernel-diz/dense_1/kernel/Initializer/random_uniform*
_output_shapes

:  *
T0*
validate_shape(*
use_locking(*%
_class
loc:@diz/dense_1/kernel
?
diz/dense_1/kernel/readIdentitydiz/dense_1/kernel*
_output_shapes

:  *
T0*%
_class
loc:@diz/dense_1/kernel
x
3diz/dense_1/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *
valueB
 *o?:
x
4diz/dense_1/kernel/Regularizer/l2_regularizer/L2LossL2Lossdiz/dense_1/kernel/read*
_output_shapes
: *
T0
?
-diz/dense_1/kernel/Regularizer/l2_regularizerMul3diz/dense_1/kernel/Regularizer/l2_regularizer/scale4diz/dense_1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
"diz/dense_1/bias/Initializer/zerosConst*
_output_shapes
: *
valueB *    *
dtype0*#
_class
loc:@diz/dense_1/bias
?
diz/dense_1/bias
VariableV2*
dtype0*
	container *
_output_shapes
: *#
_class
loc:@diz/dense_1/bias*
shared_name *
shape: 
?
diz/dense_1/bias/AssignAssigndiz/dense_1/bias"diz/dense_1/bias/Initializer/zeros*
validate_shape(*#
_class
loc:@diz/dense_1/bias*
T0*
use_locking(*
_output_shapes
: 
}
diz/dense_1/bias/readIdentitydiz/dense_1/bias*
_output_shapes
: *
T0*#
_class
loc:@diz/dense_1/bias
v
1diz/dense_1/bias/Regularizer/l2_regularizer/scaleConst*
dtype0*
valueB
 *o?:*
_output_shapes
: 
t
2diz/dense_1/bias/Regularizer/l2_regularizer/L2LossL2Lossdiz/dense_1/bias/read*
_output_shapes
: *
T0
?
+diz/dense_1/bias/Regularizer/l2_regularizerMul1diz/dense_1/bias/Regularizer/l2_regularizer/scale2diz/dense_1/bias/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
diz/dense_1/MatMulMatMuldiz/dense/LeakyReludiz/dense_1/kernel/read*
T0*
transpose_b( *'
_output_shapes
:????????? *
transpose_a( 
?
diz/dense_1/BiasAddBiasAdddiz/dense_1/MatMuldiz/dense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:????????? 
y
diz/dense_1/LeakyRelu	LeakyReludiz/dense_1/BiasAdd*
T0*
alpha%??L>*'
_output_shapes
:????????? 
?
3diz/dense_2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
_class
loc:@diz/dense_2/kernel*
valueB"        *
dtype0
?
1diz/dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *%
_class
loc:@diz/dense_2/kernel*
valueB
 *qĜ?
?
1diz/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *qĜ>*
_output_shapes
: *
dtype0*%
_class
loc:@diz/dense_2/kernel
?
;diz/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform3diz/dense_2/kernel/Initializer/random_uniform/shape*%
_class
loc:@diz/dense_2/kernel*
T0*
dtype0*
seed2 *

seed *
_output_shapes

:  
?
1diz/dense_2/kernel/Initializer/random_uniform/subSub1diz/dense_2/kernel/Initializer/random_uniform/max1diz/dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *%
_class
loc:@diz/dense_2/kernel
?
1diz/dense_2/kernel/Initializer/random_uniform/mulMul;diz/dense_2/kernel/Initializer/random_uniform/RandomUniform1diz/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes

:  *
T0*%
_class
loc:@diz/dense_2/kernel
?
-diz/dense_2/kernel/Initializer/random_uniformAdd1diz/dense_2/kernel/Initializer/random_uniform/mul1diz/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes

:  *%
_class
loc:@diz/dense_2/kernel*
T0
?
diz/dense_2/kernel
VariableV2*
shape
:  *
dtype0*
	container *%
_class
loc:@diz/dense_2/kernel*
shared_name *
_output_shapes

:  
?
diz/dense_2/kernel/AssignAssigndiz/dense_2/kernel-diz/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes

:  *
T0*%
_class
loc:@diz/dense_2/kernel*
use_locking(
?
diz/dense_2/kernel/readIdentitydiz/dense_2/kernel*
_output_shapes

:  *%
_class
loc:@diz/dense_2/kernel*
T0
x
3diz/dense_2/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
valueB
 *o?:*
_output_shapes
: 
x
4diz/dense_2/kernel/Regularizer/l2_regularizer/L2LossL2Lossdiz/dense_2/kernel/read*
_output_shapes
: *
T0
?
-diz/dense_2/kernel/Regularizer/l2_regularizerMul3diz/dense_2/kernel/Regularizer/l2_regularizer/scale4diz/dense_2/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
"diz/dense_2/bias/Initializer/zerosConst*
_output_shapes
: *#
_class
loc:@diz/dense_2/bias*
dtype0*
valueB *    
?
diz/dense_2/bias
VariableV2*
	container *
shared_name *
shape: *
_output_shapes
: *#
_class
loc:@diz/dense_2/bias*
dtype0
?
diz/dense_2/bias/AssignAssigndiz/dense_2/bias"diz/dense_2/bias/Initializer/zeros*#
_class
loc:@diz/dense_2/bias*
T0*
validate_shape(*
_output_shapes
: *
use_locking(
}
diz/dense_2/bias/readIdentitydiz/dense_2/bias*#
_class
loc:@diz/dense_2/bias*
_output_shapes
: *
T0
v
1diz/dense_2/bias/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
valueB
 *o?:*
dtype0
t
2diz/dense_2/bias/Regularizer/l2_regularizer/L2LossL2Lossdiz/dense_2/bias/read*
_output_shapes
: *
T0
?
+diz/dense_2/bias/Regularizer/l2_regularizerMul1diz/dense_2/bias/Regularizer/l2_regularizer/scale2diz/dense_2/bias/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
diz/dense_2/MatMulMatMuldiz/dense_1/LeakyReludiz/dense_2/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:????????? 
?
diz/dense_2/BiasAddBiasAdddiz/dense_2/MatMuldiz/dense_2/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:????????? 
y
diz/dense_2/LeakyRelu	LeakyReludiz/dense_2/BiasAdd*
alpha%??L>*
T0*'
_output_shapes
:????????? 
?
3diz/dense_3/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB"       *%
_class
loc:@diz/dense_3/kernel
?
1diz/dense_3/kernel/Initializer/random_uniform/minConst*
dtype0*%
_class
loc:@diz/dense_3/kernel*
_output_shapes
: *
valueB
 *JQھ
?
1diz/dense_3/kernel/Initializer/random_uniform/maxConst*
dtype0*%
_class
loc:@diz/dense_3/kernel*
_output_shapes
: *
valueB
 *JQ?>
?
;diz/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform3diz/dense_3/kernel/Initializer/random_uniform/shape*
T0*
dtype0*%
_class
loc:@diz/dense_3/kernel*
seed2 *

seed *
_output_shapes

: 
?
1diz/dense_3/kernel/Initializer/random_uniform/subSub1diz/dense_3/kernel/Initializer/random_uniform/max1diz/dense_3/kernel/Initializer/random_uniform/min*%
_class
loc:@diz/dense_3/kernel*
T0*
_output_shapes
: 
?
1diz/dense_3/kernel/Initializer/random_uniform/mulMul;diz/dense_3/kernel/Initializer/random_uniform/RandomUniform1diz/dense_3/kernel/Initializer/random_uniform/sub*%
_class
loc:@diz/dense_3/kernel*
T0*
_output_shapes

: 
?
-diz/dense_3/kernel/Initializer/random_uniformAdd1diz/dense_3/kernel/Initializer/random_uniform/mul1diz/dense_3/kernel/Initializer/random_uniform/min*%
_class
loc:@diz/dense_3/kernel*
T0*
_output_shapes

: 
?
diz/dense_3/kernel
VariableV2*%
_class
loc:@diz/dense_3/kernel*
_output_shapes

: *
shared_name *
shape
: *
dtype0*
	container 
?
diz/dense_3/kernel/AssignAssigndiz/dense_3/kernel-diz/dense_3/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes

: *%
_class
loc:@diz/dense_3/kernel*
use_locking(*
T0
?
diz/dense_3/kernel/readIdentitydiz/dense_3/kernel*
_output_shapes

: *%
_class
loc:@diz/dense_3/kernel*
T0
x
3diz/dense_3/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o?:*
dtype0*
_output_shapes
: 
x
4diz/dense_3/kernel/Regularizer/l2_regularizer/L2LossL2Lossdiz/dense_3/kernel/read*
T0*
_output_shapes
: 
?
-diz/dense_3/kernel/Regularizer/l2_regularizerMul3diz/dense_3/kernel/Regularizer/l2_regularizer/scale4diz/dense_3/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
"diz/dense_3/bias/Initializer/zerosConst*
valueB*    *
dtype0*#
_class
loc:@diz/dense_3/bias*
_output_shapes
:
?
diz/dense_3/bias
VariableV2*
_output_shapes
:*
	container *
dtype0*
shared_name *
shape:*#
_class
loc:@diz/dense_3/bias
?
diz/dense_3/bias/AssignAssigndiz/dense_3/bias"diz/dense_3/bias/Initializer/zeros*
T0*
_output_shapes
:*
use_locking(*#
_class
loc:@diz/dense_3/bias*
validate_shape(
}
diz/dense_3/bias/readIdentitydiz/dense_3/bias*
T0*#
_class
loc:@diz/dense_3/bias*
_output_shapes
:
v
1diz/dense_3/bias/Regularizer/l2_regularizer/scaleConst*
dtype0*
valueB
 *o?:*
_output_shapes
: 
t
2diz/dense_3/bias/Regularizer/l2_regularizer/L2LossL2Lossdiz/dense_3/bias/read*
_output_shapes
: *
T0
?
+diz/dense_3/bias/Regularizer/l2_regularizerMul1diz/dense_3/bias/Regularizer/l2_regularizer/scale2diz/dense_3/bias/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
diz/dense_3/MatMulMatMuldiz/dense_2/LeakyReludiz/dense_3/kernel/read*
transpose_a( *'
_output_shapes
:?????????*
T0*
transpose_b( 
?
diz/dense_3/BiasAddBiasAdddiz/dense_3/MatMuldiz/dense_3/bias/read*'
_output_shapes
:?????????*
T0*
data_formatNHWC
Y
SigmoidSigmoiddiz/dense_3/BiasAdd*'
_output_shapes
:?????????*
T0
?
diz_1/dense/MatMulMatMul@MultivariateNormalDiag/sample/affine_linear_operator/forward/adddiz/dense/kernel/read*
transpose_a( *'
_output_shapes
:????????? *
T0*
transpose_b( 
?
diz_1/dense/BiasAddBiasAdddiz_1/dense/MatMuldiz/dense/bias/read*'
_output_shapes
:????????? *
data_formatNHWC*
T0
y
diz_1/dense/LeakyRelu	LeakyReludiz_1/dense/BiasAdd*
T0*'
_output_shapes
:????????? *
alpha%??L>
?
diz_1/dense_1/MatMulMatMuldiz_1/dense/LeakyReludiz/dense_1/kernel/read*'
_output_shapes
:????????? *
transpose_a( *
T0*
transpose_b( 
?
diz_1/dense_1/BiasAddBiasAdddiz_1/dense_1/MatMuldiz/dense_1/bias/read*'
_output_shapes
:????????? *
data_formatNHWC*
T0
}
diz_1/dense_1/LeakyRelu	LeakyReludiz_1/dense_1/BiasAdd*
T0*'
_output_shapes
:????????? *
alpha%??L>
?
diz_1/dense_2/MatMulMatMuldiz_1/dense_1/LeakyReludiz/dense_2/kernel/read*'
_output_shapes
:????????? *
transpose_b( *
T0*
transpose_a( 
?
diz_1/dense_2/BiasAddBiasAdddiz_1/dense_2/MatMuldiz/dense_2/bias/read*'
_output_shapes
:????????? *
data_formatNHWC*
T0
}
diz_1/dense_2/LeakyRelu	LeakyReludiz_1/dense_2/BiasAdd*'
_output_shapes
:????????? *
T0*
alpha%??L>
?
diz_1/dense_3/MatMulMatMuldiz_1/dense_2/LeakyReludiz/dense_3/kernel/read*'
_output_shapes
:?????????*
transpose_a( *
transpose_b( *
T0
?
diz_1/dense_3/BiasAddBiasAdddiz_1/dense_3/MatMuldiz/dense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:?????????
]
	Sigmoid_1Sigmoiddiz_1/dense_3/BiasAdd*'
_output_shapes
:?????????*
T0
n
decoder/Reshape/shapeConst*
_output_shapes
:*%
valueB"????      d   *
dtype0
?
decoder/ReshapeReshapeHencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/adddecoder/Reshape/shape*/
_output_shapes
:?????????d*
T0*
Tshape0
?
9decoder/layer_0/kernel/Initializer/truncated_normal/shapeConst*
dtype0*)
_class
loc:@decoder/layer_0/kernel*%
valueB"         d   *
_output_shapes
:
?
8decoder/layer_0/kernel/Initializer/truncated_normal/meanConst*)
_class
loc:@decoder/layer_0/kernel*
_output_shapes
: *
valueB
 *    *
dtype0
?
:decoder/layer_0/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
ף<*)
_class
loc:@decoder/layer_0/kernel*
_output_shapes
: *
dtype0
?
Cdecoder/layer_0/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9decoder/layer_0/kernel/Initializer/truncated_normal/shape*

seed *
T0*
dtype0*)
_class
loc:@decoder/layer_0/kernel*'
_output_shapes
:?d*
seed2 
?
7decoder/layer_0/kernel/Initializer/truncated_normal/mulMulCdecoder/layer_0/kernel/Initializer/truncated_normal/TruncatedNormal:decoder/layer_0/kernel/Initializer/truncated_normal/stddev*)
_class
loc:@decoder/layer_0/kernel*'
_output_shapes
:?d*
T0
?
3decoder/layer_0/kernel/Initializer/truncated_normalAdd7decoder/layer_0/kernel/Initializer/truncated_normal/mul8decoder/layer_0/kernel/Initializer/truncated_normal/mean*)
_class
loc:@decoder/layer_0/kernel*
T0*'
_output_shapes
:?d
?
decoder/layer_0/kernel
VariableV2*
dtype0*)
_class
loc:@decoder/layer_0/kernel*
shape:?d*
	container *'
_output_shapes
:?d*
shared_name 
?
decoder/layer_0/kernel/AssignAssigndecoder/layer_0/kernel3decoder/layer_0/kernel/Initializer/truncated_normal*'
_output_shapes
:?d*
use_locking(*)
_class
loc:@decoder/layer_0/kernel*
validate_shape(*
T0
?
decoder/layer_0/kernel/readIdentitydecoder/layer_0/kernel*
T0*)
_class
loc:@decoder/layer_0/kernel*'
_output_shapes
:?d
?
&decoder/layer_0/bias/Initializer/zerosConst*
dtype0*'
_class
loc:@decoder/layer_0/bias*
_output_shapes	
:?*
valueB?*    
?
decoder/layer_0/bias
VariableV2*
_output_shapes	
:?*
shared_name *
dtype0*
shape:?*
	container *'
_class
loc:@decoder/layer_0/bias
?
decoder/layer_0/bias/AssignAssigndecoder/layer_0/bias&decoder/layer_0/bias/Initializer/zeros*
validate_shape(*
T0*'
_class
loc:@decoder/layer_0/bias*
use_locking(*
_output_shapes	
:?
?
decoder/layer_0/bias/readIdentitydecoder/layer_0/bias*
_output_shapes	
:?*'
_class
loc:@decoder/layer_0/bias*
T0
d
decoder/layer_0/ShapeShapedecoder/Reshape*
T0*
_output_shapes
:*
out_type0
m
#decoder/layer_0/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%decoder/layer_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
o
%decoder/layer_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
decoder/layer_0/strided_sliceStridedSlicedecoder/layer_0/Shape#decoder/layer_0/strided_slice/stack%decoder/layer_0/strided_slice/stack_1%decoder/layer_0/strided_slice/stack_2*
new_axis_mask *
ellipsis_mask *
end_mask *
shrink_axis_mask*
_output_shapes
: *

begin_mask *
Index0*
T0
o
%decoder/layer_0/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
q
'decoder/layer_0/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
q
'decoder/layer_0/strided_slice_1/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
decoder/layer_0/strided_slice_1StridedSlicedecoder/layer_0/Shape%decoder/layer_0/strided_slice_1/stack'decoder/layer_0/strided_slice_1/stack_1'decoder/layer_0/strided_slice_1/stack_2*
new_axis_mask *
_output_shapes
: *
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
T0*
end_mask 
o
%decoder/layer_0/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
q
'decoder/layer_0/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
q
'decoder/layer_0/strided_slice_2/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
?
decoder/layer_0/strided_slice_2StridedSlicedecoder/layer_0/Shape%decoder/layer_0/strided_slice_2/stack'decoder/layer_0/strided_slice_2/stack_1'decoder/layer_0/strided_slice_2/stack_2*
shrink_axis_mask*
ellipsis_mask *
Index0*
end_mask *
T0*
new_axis_mask *

begin_mask *
_output_shapes
: 
W
decoder/layer_0/mul/yConst*
_output_shapes
: *
value	B :*
dtype0
s
decoder/layer_0/mulMuldecoder/layer_0/strided_slice_1decoder/layer_0/mul/y*
_output_shapes
: *
T0
W
decoder/layer_0/add/yConst*
value	B :*
_output_shapes
: *
dtype0
g
decoder/layer_0/addAdddecoder/layer_0/muldecoder/layer_0/add/y*
_output_shapes
: *
T0
Y
decoder/layer_0/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
w
decoder/layer_0/mul_1Muldecoder/layer_0/strided_slice_2decoder/layer_0/mul_1/y*
T0*
_output_shapes
: 
Y
decoder/layer_0/add_1/yConst*
_output_shapes
: *
value	B :*
dtype0
m
decoder/layer_0/add_1Adddecoder/layer_0/mul_1decoder/layer_0/add_1/y*
_output_shapes
: *
T0
Z
decoder/layer_0/stack/3Const*
dtype0*
_output_shapes
: *
value
B :?
?
decoder/layer_0/stackPackdecoder/layer_0/strided_slicedecoder/layer_0/adddecoder/layer_0/add_1decoder/layer_0/stack/3*
N*
_output_shapes
:*
T0*

axis 
o
%decoder/layer_0/strided_slice_3/stackConst*
valueB: *
_output_shapes
:*
dtype0
q
'decoder/layer_0/strided_slice_3/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
q
'decoder/layer_0/strided_slice_3/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
decoder/layer_0/strided_slice_3StridedSlicedecoder/layer_0/stack%decoder/layer_0/strided_slice_3/stack'decoder/layer_0/strided_slice_3/stack_1'decoder/layer_0/strided_slice_3/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
Index0*
_output_shapes
: *
T0*
new_axis_mask *
end_mask 
?
 decoder/layer_0/conv2d_transposeConv2DBackpropInputdecoder/layer_0/stackdecoder/layer_0/kernel/readdecoder/Reshape*0
_output_shapes
:??????????*
paddingVALID*
data_formatNHWC*
use_cudnn_on_gpu(*
T0*
strides
*
explicit_paddings
 *
	dilations

?
decoder/layer_0/BiasAddBiasAdd decoder/layer_0/conv2d_transposedecoder/layer_0/bias/read*
T0*
data_formatNHWC*0
_output_shapes
:??????????
?
2decoder/batch_normalization/gamma/Initializer/onesConst*
valueB?*  ??*4
_class*
(&loc:@decoder/batch_normalization/gamma*
_output_shapes	
:?*
dtype0
?
!decoder/batch_normalization/gamma
VariableV2*
	container *
shared_name *
shape:?*
_output_shapes	
:?*4
_class*
(&loc:@decoder/batch_normalization/gamma*
dtype0
?
(decoder/batch_normalization/gamma/AssignAssign!decoder/batch_normalization/gamma2decoder/batch_normalization/gamma/Initializer/ones*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(*4
_class*
(&loc:@decoder/batch_normalization/gamma
?
&decoder/batch_normalization/gamma/readIdentity!decoder/batch_normalization/gamma*
T0*
_output_shapes	
:?*4
_class*
(&loc:@decoder/batch_normalization/gamma
?
2decoder/batch_normalization/beta/Initializer/zerosConst*
dtype0*3
_class)
'%loc:@decoder/batch_normalization/beta*
_output_shapes	
:?*
valueB?*    
?
 decoder/batch_normalization/beta
VariableV2*
dtype0*3
_class)
'%loc:@decoder/batch_normalization/beta*
_output_shapes	
:?*
shape:?*
	container *
shared_name 
?
'decoder/batch_normalization/beta/AssignAssign decoder/batch_normalization/beta2decoder/batch_normalization/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*3
_class)
'%loc:@decoder/batch_normalization/beta*
T0
?
%decoder/batch_normalization/beta/readIdentity decoder/batch_normalization/beta*
_output_shapes	
:?*
T0*3
_class)
'%loc:@decoder/batch_normalization/beta
?
9decoder/batch_normalization/moving_mean/Initializer/zerosConst*
valueB?*    *:
_class0
.,loc:@decoder/batch_normalization/moving_mean*
dtype0*
_output_shapes	
:?
?
'decoder/batch_normalization/moving_mean
VariableV2*
shared_name *
dtype0*:
_class0
.,loc:@decoder/batch_normalization/moving_mean*
shape:?*
	container *
_output_shapes	
:?
?
.decoder/batch_normalization/moving_mean/AssignAssign'decoder/batch_normalization/moving_mean9decoder/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*:
_class0
.,loc:@decoder/batch_normalization/moving_mean*
T0*
validate_shape(*
_output_shapes	
:?
?
,decoder/batch_normalization/moving_mean/readIdentity'decoder/batch_normalization/moving_mean*
_output_shapes	
:?*:
_class0
.,loc:@decoder/batch_normalization/moving_mean*
T0
?
<decoder/batch_normalization/moving_variance/Initializer/onesConst*
valueB?*  ??*
dtype0*
_output_shapes	
:?*>
_class4
20loc:@decoder/batch_normalization/moving_variance
?
+decoder/batch_normalization/moving_variance
VariableV2*
shape:?*
_output_shapes	
:?*
dtype0*>
_class4
20loc:@decoder/batch_normalization/moving_variance*
	container *
shared_name 
?
2decoder/batch_normalization/moving_variance/AssignAssign+decoder/batch_normalization/moving_variance<decoder/batch_normalization/moving_variance/Initializer/ones*
T0*>
_class4
20loc:@decoder/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
0decoder/batch_normalization/moving_variance/readIdentity+decoder/batch_normalization/moving_variance*
T0*
_output_shapes	
:?*>
_class4
20loc:@decoder/batch_normalization/moving_variance
d
!decoder/batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB 
f
#decoder/batch_normalization/Const_1Const*
_output_shapes
: *
valueB *
dtype0
?
*decoder/batch_normalization/FusedBatchNormFusedBatchNormdecoder/layer_0/BiasAdd&decoder/batch_normalization/gamma/read%decoder/batch_normalization/beta/read!decoder/batch_normalization/Const#decoder/batch_normalization/Const_1*
is_training(*
epsilon%o?:*L
_output_shapes:
8:??????????:?:?:?:?*
data_formatNHWC*
T0
h
#decoder/batch_normalization/Const_2Const*
dtype0*
valueB
 *?p}?*
_output_shapes
: 
?
1decoder/batch_normalization/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*:
_class0
.,loc:@decoder/batch_normalization/moving_mean*
valueB
 *  ??
?
/decoder/batch_normalization/AssignMovingAvg/subSub1decoder/batch_normalization/AssignMovingAvg/sub/x#decoder/batch_normalization/Const_2*
_output_shapes
: *:
_class0
.,loc:@decoder/batch_normalization/moving_mean*
T0
?
1decoder/batch_normalization/AssignMovingAvg/sub_1Sub,decoder/batch_normalization/moving_mean/read,decoder/batch_normalization/FusedBatchNorm:1*
_output_shapes	
:?*
T0*:
_class0
.,loc:@decoder/batch_normalization/moving_mean
?
/decoder/batch_normalization/AssignMovingAvg/mulMul1decoder/batch_normalization/AssignMovingAvg/sub_1/decoder/batch_normalization/AssignMovingAvg/sub*
_output_shapes	
:?*:
_class0
.,loc:@decoder/batch_normalization/moving_mean*
T0
?
+decoder/batch_normalization/AssignMovingAvg	AssignSub'decoder/batch_normalization/moving_mean/decoder/batch_normalization/AssignMovingAvg/mul*
_output_shapes	
:?*:
_class0
.,loc:@decoder/batch_normalization/moving_mean*
T0*
use_locking( 
?
3decoder/batch_normalization/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*>
_class4
20loc:@decoder/batch_normalization/moving_variance*
valueB
 *  ??
?
1decoder/batch_normalization/AssignMovingAvg_1/subSub3decoder/batch_normalization/AssignMovingAvg_1/sub/x#decoder/batch_normalization/Const_2*
T0*>
_class4
20loc:@decoder/batch_normalization/moving_variance*
_output_shapes
: 
?
3decoder/batch_normalization/AssignMovingAvg_1/sub_1Sub0decoder/batch_normalization/moving_variance/read,decoder/batch_normalization/FusedBatchNorm:2*>
_class4
20loc:@decoder/batch_normalization/moving_variance*
T0*
_output_shapes	
:?
?
1decoder/batch_normalization/AssignMovingAvg_1/mulMul3decoder/batch_normalization/AssignMovingAvg_1/sub_11decoder/batch_normalization/AssignMovingAvg_1/sub*>
_class4
20loc:@decoder/batch_normalization/moving_variance*
_output_shapes	
:?*
T0
?
-decoder/batch_normalization/AssignMovingAvg_1	AssignSub+decoder/batch_normalization/moving_variance1decoder/batch_normalization/AssignMovingAvg_1/mul*>
_class4
20loc:@decoder/batch_normalization/moving_variance*
use_locking( *
_output_shapes	
:?*
T0
?
decoder/LeakyRelu	LeakyRelu*decoder/batch_normalization/FusedBatchNorm*0
_output_shapes
:??????????*
T0*
alpha%??L>
?
9decoder/layer_1/kernel/Initializer/truncated_normal/shapeConst*%
valueB"      ?      *
dtype0*
_output_shapes
:*)
_class
loc:@decoder/layer_1/kernel
?
8decoder/layer_1/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *)
_class
loc:@decoder/layer_1/kernel*
dtype0*
_output_shapes
: 
?
:decoder/layer_1/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<*)
_class
loc:@decoder/layer_1/kernel
?
Cdecoder/layer_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9decoder/layer_1/kernel/Initializer/truncated_normal/shape*
T0*
seed2 *)
_class
loc:@decoder/layer_1/kernel*

seed *
dtype0*(
_output_shapes
:??
?
7decoder/layer_1/kernel/Initializer/truncated_normal/mulMulCdecoder/layer_1/kernel/Initializer/truncated_normal/TruncatedNormal:decoder/layer_1/kernel/Initializer/truncated_normal/stddev*)
_class
loc:@decoder/layer_1/kernel*
T0*(
_output_shapes
:??
?
3decoder/layer_1/kernel/Initializer/truncated_normalAdd7decoder/layer_1/kernel/Initializer/truncated_normal/mul8decoder/layer_1/kernel/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*)
_class
loc:@decoder/layer_1/kernel
?
decoder/layer_1/kernel
VariableV2*
	container *(
_output_shapes
:??*
shape:??*)
_class
loc:@decoder/layer_1/kernel*
dtype0*
shared_name 
?
decoder/layer_1/kernel/AssignAssigndecoder/layer_1/kernel3decoder/layer_1/kernel/Initializer/truncated_normal*)
_class
loc:@decoder/layer_1/kernel*
T0*
use_locking(*
validate_shape(*(
_output_shapes
:??
?
decoder/layer_1/kernel/readIdentitydecoder/layer_1/kernel*)
_class
loc:@decoder/layer_1/kernel*(
_output_shapes
:??*
T0
?
&decoder/layer_1/bias/Initializer/zerosConst*
_output_shapes	
:?*
dtype0*'
_class
loc:@decoder/layer_1/bias*
valueB?*    
?
decoder/layer_1/bias
VariableV2*'
_class
loc:@decoder/layer_1/bias*
_output_shapes	
:?*
dtype0*
shape:?*
	container *
shared_name 
?
decoder/layer_1/bias/AssignAssigndecoder/layer_1/bias&decoder/layer_1/bias/Initializer/zeros*
T0*
validate_shape(*
_output_shapes	
:?*
use_locking(*'
_class
loc:@decoder/layer_1/bias
?
decoder/layer_1/bias/readIdentitydecoder/layer_1/bias*
_output_shapes	
:?*
T0*'
_class
loc:@decoder/layer_1/bias
f
decoder/layer_1/ShapeShapedecoder/LeakyRelu*
T0*
_output_shapes
:*
out_type0
m
#decoder/layer_1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
o
%decoder/layer_1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
o
%decoder/layer_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
decoder/layer_1/strided_sliceStridedSlicedecoder/layer_1/Shape#decoder/layer_1/strided_slice/stack%decoder/layer_1/strided_slice/stack_1%decoder/layer_1/strided_slice/stack_2*
new_axis_mask *
_output_shapes
: *
shrink_axis_mask*

begin_mask *
T0*
end_mask *
ellipsis_mask *
Index0
o
%decoder/layer_1/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
q
'decoder/layer_1/strided_slice_1/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
q
'decoder/layer_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
decoder/layer_1/strided_slice_1StridedSlicedecoder/layer_1/Shape%decoder/layer_1/strided_slice_1/stack'decoder/layer_1/strided_slice_1/stack_1'decoder/layer_1/strided_slice_1/stack_2*
end_mask *

begin_mask *
shrink_axis_mask*
new_axis_mask *
_output_shapes
: *
ellipsis_mask *
T0*
Index0
o
%decoder/layer_1/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
q
'decoder/layer_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
q
'decoder/layer_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
decoder/layer_1/strided_slice_2StridedSlicedecoder/layer_1/Shape%decoder/layer_1/strided_slice_2/stack'decoder/layer_1/strided_slice_2/stack_1'decoder/layer_1/strided_slice_2/stack_2*
end_mask *
ellipsis_mask *

begin_mask *
T0*
shrink_axis_mask*
_output_shapes
: *
new_axis_mask *
Index0
W
decoder/layer_1/mul/yConst*
dtype0*
_output_shapes
: *
value	B :
s
decoder/layer_1/mulMuldecoder/layer_1/strided_slice_1decoder/layer_1/mul/y*
T0*
_output_shapes
: 
Y
decoder/layer_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :
w
decoder/layer_1/mul_1Muldecoder/layer_1/strided_slice_2decoder/layer_1/mul_1/y*
_output_shapes
: *
T0
Z
decoder/layer_1/stack/3Const*
dtype0*
_output_shapes
: *
value
B :?
?
decoder/layer_1/stackPackdecoder/layer_1/strided_slicedecoder/layer_1/muldecoder/layer_1/mul_1decoder/layer_1/stack/3*
_output_shapes
:*
T0*

axis *
N
o
%decoder/layer_1/strided_slice_3/stackConst*
dtype0*
valueB: *
_output_shapes
:
q
'decoder/layer_1/strided_slice_3/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
q
'decoder/layer_1/strided_slice_3/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
decoder/layer_1/strided_slice_3StridedSlicedecoder/layer_1/stack%decoder/layer_1/strided_slice_3/stack'decoder/layer_1/strided_slice_3/stack_1'decoder/layer_1/strided_slice_3/stack_2*
shrink_axis_mask*
_output_shapes
: *
end_mask *
Index0*
T0*

begin_mask *
ellipsis_mask *
new_axis_mask 
?
 decoder/layer_1/conv2d_transposeConv2DBackpropInputdecoder/layer_1/stackdecoder/layer_1/kernel/readdecoder/LeakyRelu*
T0*
paddingSAME*
explicit_paddings
 *
data_formatNHWC*
use_cudnn_on_gpu(*0
_output_shapes
:??????????*
strides
*
	dilations

?
decoder/layer_1/BiasAddBiasAdd decoder/layer_1/conv2d_transposedecoder/layer_1/bias/read*0
_output_shapes
:??????????*
T0*
data_formatNHWC
?
4decoder/batch_normalization_1/gamma/Initializer/onesConst*
valueB?*  ??*
dtype0*
_output_shapes	
:?*6
_class,
*(loc:@decoder/batch_normalization_1/gamma
?
#decoder/batch_normalization_1/gamma
VariableV2*
	container *
shared_name *6
_class,
*(loc:@decoder/batch_normalization_1/gamma*
shape:?*
_output_shapes	
:?*
dtype0
?
*decoder/batch_normalization_1/gamma/AssignAssign#decoder/batch_normalization_1/gamma4decoder/batch_normalization_1/gamma/Initializer/ones*6
_class,
*(loc:@decoder/batch_normalization_1/gamma*
T0*
_output_shapes	
:?*
validate_shape(*
use_locking(
?
(decoder/batch_normalization_1/gamma/readIdentity#decoder/batch_normalization_1/gamma*
_output_shapes	
:?*6
_class,
*(loc:@decoder/batch_normalization_1/gamma*
T0
?
4decoder/batch_normalization_1/beta/Initializer/zerosConst*5
_class+
)'loc:@decoder/batch_normalization_1/beta*
dtype0*
valueB?*    *
_output_shapes	
:?
?
"decoder/batch_normalization_1/beta
VariableV2*
	container *
shape:?*5
_class+
)'loc:@decoder/batch_normalization_1/beta*
_output_shapes	
:?*
shared_name *
dtype0
?
)decoder/batch_normalization_1/beta/AssignAssign"decoder/batch_normalization_1/beta4decoder/batch_normalization_1/beta/Initializer/zeros*
use_locking(*5
_class+
)'loc:@decoder/batch_normalization_1/beta*
T0*
_output_shapes	
:?*
validate_shape(
?
'decoder/batch_normalization_1/beta/readIdentity"decoder/batch_normalization_1/beta*
_output_shapes	
:?*5
_class+
)'loc:@decoder/batch_normalization_1/beta*
T0
?
;decoder/batch_normalization_1/moving_mean/Initializer/zerosConst*
_output_shapes	
:?*
dtype0*<
_class2
0.loc:@decoder/batch_normalization_1/moving_mean*
valueB?*    
?
)decoder/batch_normalization_1/moving_mean
VariableV2*
shape:?*
shared_name *<
_class2
0.loc:@decoder/batch_normalization_1/moving_mean*
_output_shapes	
:?*
dtype0*
	container 
?
0decoder/batch_normalization_1/moving_mean/AssignAssign)decoder/batch_normalization_1/moving_mean;decoder/batch_normalization_1/moving_mean/Initializer/zeros*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?*<
_class2
0.loc:@decoder/batch_normalization_1/moving_mean
?
.decoder/batch_normalization_1/moving_mean/readIdentity)decoder/batch_normalization_1/moving_mean*
_output_shapes	
:?*
T0*<
_class2
0.loc:@decoder/batch_normalization_1/moving_mean
?
>decoder/batch_normalization_1/moving_variance/Initializer/onesConst*@
_class6
42loc:@decoder/batch_normalization_1/moving_variance*
_output_shapes	
:?*
valueB?*  ??*
dtype0
?
-decoder/batch_normalization_1/moving_variance
VariableV2*
dtype0*
shared_name *
	container *
shape:?*@
_class6
42loc:@decoder/batch_normalization_1/moving_variance*
_output_shapes	
:?
?
4decoder/batch_normalization_1/moving_variance/AssignAssign-decoder/batch_normalization_1/moving_variance>decoder/batch_normalization_1/moving_variance/Initializer/ones*
T0*
use_locking(*@
_class6
42loc:@decoder/batch_normalization_1/moving_variance*
_output_shapes	
:?*
validate_shape(
?
2decoder/batch_normalization_1/moving_variance/readIdentity-decoder/batch_normalization_1/moving_variance*
_output_shapes	
:?*@
_class6
42loc:@decoder/batch_normalization_1/moving_variance*
T0
f
#decoder/batch_normalization_1/ConstConst*
valueB *
_output_shapes
: *
dtype0
h
%decoder/batch_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
?
,decoder/batch_normalization_1/FusedBatchNormFusedBatchNormdecoder/layer_1/BiasAdd(decoder/batch_normalization_1/gamma/read'decoder/batch_normalization_1/beta/read#decoder/batch_normalization_1/Const%decoder/batch_normalization_1/Const_1*
T0*
epsilon%o?:*
is_training(*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?
j
%decoder/batch_normalization_1/Const_2Const*
valueB
 *?p}?*
dtype0*
_output_shapes
: 
?
3decoder/batch_normalization_1/AssignMovingAvg/sub/xConst*
dtype0*
_output_shapes
: *<
_class2
0.loc:@decoder/batch_normalization_1/moving_mean*
valueB
 *  ??
?
1decoder/batch_normalization_1/AssignMovingAvg/subSub3decoder/batch_normalization_1/AssignMovingAvg/sub/x%decoder/batch_normalization_1/Const_2*
_output_shapes
: *
T0*<
_class2
0.loc:@decoder/batch_normalization_1/moving_mean
?
3decoder/batch_normalization_1/AssignMovingAvg/sub_1Sub.decoder/batch_normalization_1/moving_mean/read.decoder/batch_normalization_1/FusedBatchNorm:1*<
_class2
0.loc:@decoder/batch_normalization_1/moving_mean*
_output_shapes	
:?*
T0
?
1decoder/batch_normalization_1/AssignMovingAvg/mulMul3decoder/batch_normalization_1/AssignMovingAvg/sub_11decoder/batch_normalization_1/AssignMovingAvg/sub*<
_class2
0.loc:@decoder/batch_normalization_1/moving_mean*
T0*
_output_shapes	
:?
?
-decoder/batch_normalization_1/AssignMovingAvg	AssignSub)decoder/batch_normalization_1/moving_mean1decoder/batch_normalization_1/AssignMovingAvg/mul*
T0*
_output_shapes	
:?*
use_locking( *<
_class2
0.loc:@decoder/batch_normalization_1/moving_mean
?
5decoder/batch_normalization_1/AssignMovingAvg_1/sub/xConst*@
_class6
42loc:@decoder/batch_normalization_1/moving_variance*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
3decoder/batch_normalization_1/AssignMovingAvg_1/subSub5decoder/batch_normalization_1/AssignMovingAvg_1/sub/x%decoder/batch_normalization_1/Const_2*@
_class6
42loc:@decoder/batch_normalization_1/moving_variance*
T0*
_output_shapes
: 
?
5decoder/batch_normalization_1/AssignMovingAvg_1/sub_1Sub2decoder/batch_normalization_1/moving_variance/read.decoder/batch_normalization_1/FusedBatchNorm:2*
T0*
_output_shapes	
:?*@
_class6
42loc:@decoder/batch_normalization_1/moving_variance
?
3decoder/batch_normalization_1/AssignMovingAvg_1/mulMul5decoder/batch_normalization_1/AssignMovingAvg_1/sub_13decoder/batch_normalization_1/AssignMovingAvg_1/sub*@
_class6
42loc:@decoder/batch_normalization_1/moving_variance*
T0*
_output_shapes	
:?
?
/decoder/batch_normalization_1/AssignMovingAvg_1	AssignSub-decoder/batch_normalization_1/moving_variance3decoder/batch_normalization_1/AssignMovingAvg_1/mul*
T0*
use_locking( *
_output_shapes	
:?*@
_class6
42loc:@decoder/batch_normalization_1/moving_variance
?
decoder/LeakyRelu_1	LeakyRelu,decoder/batch_normalization_1/FusedBatchNorm*
T0*
alpha%??L>*0
_output_shapes
:??????????
?
9decoder/layer_2/kernel/Initializer/truncated_normal/shapeConst*%
valueB"         ?   *
dtype0*
_output_shapes
:*)
_class
loc:@decoder/layer_2/kernel
?
8decoder/layer_2/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: *)
_class
loc:@decoder/layer_2/kernel
?
:decoder/layer_2/kernel/Initializer/truncated_normal/stddevConst*)
_class
loc:@decoder/layer_2/kernel*
valueB
 *
ף<*
_output_shapes
: *
dtype0
?
Cdecoder/layer_2/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9decoder/layer_2/kernel/Initializer/truncated_normal/shape*
dtype0*'
_output_shapes
:?*
seed2 *
T0*)
_class
loc:@decoder/layer_2/kernel*

seed 
?
7decoder/layer_2/kernel/Initializer/truncated_normal/mulMulCdecoder/layer_2/kernel/Initializer/truncated_normal/TruncatedNormal:decoder/layer_2/kernel/Initializer/truncated_normal/stddev*
T0*)
_class
loc:@decoder/layer_2/kernel*'
_output_shapes
:?
?
3decoder/layer_2/kernel/Initializer/truncated_normalAdd7decoder/layer_2/kernel/Initializer/truncated_normal/mul8decoder/layer_2/kernel/Initializer/truncated_normal/mean*
T0*)
_class
loc:@decoder/layer_2/kernel*'
_output_shapes
:?
?
decoder/layer_2/kernel
VariableV2*
	container *
shared_name *)
_class
loc:@decoder/layer_2/kernel*
shape:?*'
_output_shapes
:?*
dtype0
?
decoder/layer_2/kernel/AssignAssigndecoder/layer_2/kernel3decoder/layer_2/kernel/Initializer/truncated_normal*'
_output_shapes
:?*)
_class
loc:@decoder/layer_2/kernel*
use_locking(*
T0*
validate_shape(
?
decoder/layer_2/kernel/readIdentitydecoder/layer_2/kernel*'
_output_shapes
:?*
T0*)
_class
loc:@decoder/layer_2/kernel
?
&decoder/layer_2/bias/Initializer/zerosConst*'
_class
loc:@decoder/layer_2/bias*
dtype0*
_output_shapes
:*
valueB*    
?
decoder/layer_2/bias
VariableV2*
	container *
_output_shapes
:*
dtype0*
shared_name *'
_class
loc:@decoder/layer_2/bias*
shape:
?
decoder/layer_2/bias/AssignAssigndecoder/layer_2/bias&decoder/layer_2/bias/Initializer/zeros*'
_class
loc:@decoder/layer_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
?
decoder/layer_2/bias/readIdentitydecoder/layer_2/bias*'
_class
loc:@decoder/layer_2/bias*
_output_shapes
:*
T0
h
decoder/layer_2/ShapeShapedecoder/LeakyRelu_1*
T0*
out_type0*
_output_shapes
:
m
#decoder/layer_2/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%decoder/layer_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%decoder/layer_2/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
decoder/layer_2/strided_sliceStridedSlicedecoder/layer_2/Shape#decoder/layer_2/strided_slice/stack%decoder/layer_2/strided_slice/stack_1%decoder/layer_2/strided_slice/stack_2*
shrink_axis_mask*
Index0*
_output_shapes
: *
end_mask *
new_axis_mask *
T0*

begin_mask *
ellipsis_mask 
o
%decoder/layer_2/strided_slice_1/stackConst*
_output_shapes
:*
valueB:*
dtype0
q
'decoder/layer_2/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
q
'decoder/layer_2/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
decoder/layer_2/strided_slice_1StridedSlicedecoder/layer_2/Shape%decoder/layer_2/strided_slice_1/stack'decoder/layer_2/strided_slice_1/stack_1'decoder/layer_2/strided_slice_1/stack_2*
new_axis_mask *
T0*

begin_mask *
Index0*
ellipsis_mask *
shrink_axis_mask*
end_mask *
_output_shapes
: 
o
%decoder/layer_2/strided_slice_2/stackConst*
dtype0*
valueB:*
_output_shapes
:
q
'decoder/layer_2/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
q
'decoder/layer_2/strided_slice_2/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
?
decoder/layer_2/strided_slice_2StridedSlicedecoder/layer_2/Shape%decoder/layer_2/strided_slice_2/stack'decoder/layer_2/strided_slice_2/stack_1'decoder/layer_2/strided_slice_2/stack_2*
shrink_axis_mask*
Index0*
_output_shapes
: *
end_mask *
T0*
ellipsis_mask *

begin_mask *
new_axis_mask 
W
decoder/layer_2/mul/yConst*
value	B :*
_output_shapes
: *
dtype0
s
decoder/layer_2/mulMuldecoder/layer_2/strided_slice_1decoder/layer_2/mul/y*
T0*
_output_shapes
: 
Y
decoder/layer_2/mul_1/yConst*
_output_shapes
: *
value	B :*
dtype0
w
decoder/layer_2/mul_1Muldecoder/layer_2/strided_slice_2decoder/layer_2/mul_1/y*
T0*
_output_shapes
: 
Y
decoder/layer_2/stack/3Const*
_output_shapes
: *
value	B :*
dtype0
?
decoder/layer_2/stackPackdecoder/layer_2/strided_slicedecoder/layer_2/muldecoder/layer_2/mul_1decoder/layer_2/stack/3*
N*
_output_shapes
:*
T0*

axis 
o
%decoder/layer_2/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
q
'decoder/layer_2/strided_slice_3/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
q
'decoder/layer_2/strided_slice_3/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
?
decoder/layer_2/strided_slice_3StridedSlicedecoder/layer_2/stack%decoder/layer_2/strided_slice_3/stack'decoder/layer_2/strided_slice_3/stack_1'decoder/layer_2/strided_slice_3/stack_2*
new_axis_mask *
shrink_axis_mask*
_output_shapes
: *
ellipsis_mask *

begin_mask *
end_mask *
Index0*
T0
?
 decoder/layer_2/conv2d_transposeConv2DBackpropInputdecoder/layer_2/stackdecoder/layer_2/kernel/readdecoder/LeakyRelu_1*
explicit_paddings
 *
	dilations
*/
_output_shapes
:?????????*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*
strides
*
T0
?
decoder/layer_2/BiasAddBiasAdd decoder/layer_2/conv2d_transposedecoder/layer_2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:?????????
o
decoder/layer_2/TanhTanhdecoder/layer_2/BiasAdd*
T0*/
_output_shapes
:?????????
t
decoder/generated_imagesIdentitydecoder/layer_2/Tanh*/
_output_shapes
:?????????*
T0
p
decoder_1/Reshape/shapeConst*
_output_shapes
:*%
valueB"????      d   *
dtype0
?
decoder_1/ReshapeReshape@MultivariateNormalDiag/sample/affine_linear_operator/forward/adddecoder_1/Reshape/shape*
Tshape0*/
_output_shapes
:?????????d*
T0
h
decoder_1/layer_0/ShapeShapedecoder_1/Reshape*
T0*
_output_shapes
:*
out_type0
o
%decoder_1/layer_0/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
q
'decoder_1/layer_0/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
q
'decoder_1/layer_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
decoder_1/layer_0/strided_sliceStridedSlicedecoder_1/layer_0/Shape%decoder_1/layer_0/strided_slice/stack'decoder_1/layer_0/strided_slice/stack_1'decoder_1/layer_0/strided_slice/stack_2*
T0*
_output_shapes
: *
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
end_mask *
new_axis_mask 
q
'decoder_1/layer_0/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
s
)decoder_1/layer_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
s
)decoder_1/layer_0/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
!decoder_1/layer_0/strided_slice_1StridedSlicedecoder_1/layer_0/Shape'decoder_1/layer_0/strided_slice_1/stack)decoder_1/layer_0/strided_slice_1/stack_1)decoder_1/layer_0/strided_slice_1/stack_2*
end_mask *
shrink_axis_mask*

begin_mask *
Index0*
T0*
new_axis_mask *
ellipsis_mask *
_output_shapes
: 
q
'decoder_1/layer_0/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
s
)decoder_1/layer_0/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
s
)decoder_1/layer_0/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
!decoder_1/layer_0/strided_slice_2StridedSlicedecoder_1/layer_0/Shape'decoder_1/layer_0/strided_slice_2/stack)decoder_1/layer_0/strided_slice_2/stack_1)decoder_1/layer_0/strided_slice_2/stack_2*
T0*
new_axis_mask *
ellipsis_mask *
shrink_axis_mask*
end_mask *
_output_shapes
: *

begin_mask *
Index0
Y
decoder_1/layer_0/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
y
decoder_1/layer_0/mulMul!decoder_1/layer_0/strided_slice_1decoder_1/layer_0/mul/y*
_output_shapes
: *
T0
Y
decoder_1/layer_0/add/yConst*
_output_shapes
: *
value	B :*
dtype0
m
decoder_1/layer_0/addAdddecoder_1/layer_0/muldecoder_1/layer_0/add/y*
T0*
_output_shapes
: 
[
decoder_1/layer_0/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
}
decoder_1/layer_0/mul_1Mul!decoder_1/layer_0/strided_slice_2decoder_1/layer_0/mul_1/y*
T0*
_output_shapes
: 
[
decoder_1/layer_0/add_1/yConst*
_output_shapes
: *
value	B :*
dtype0
s
decoder_1/layer_0/add_1Adddecoder_1/layer_0/mul_1decoder_1/layer_0/add_1/y*
_output_shapes
: *
T0
\
decoder_1/layer_0/stack/3Const*
value
B :?*
_output_shapes
: *
dtype0
?
decoder_1/layer_0/stackPackdecoder_1/layer_0/strided_slicedecoder_1/layer_0/adddecoder_1/layer_0/add_1decoder_1/layer_0/stack/3*
N*
T0*
_output_shapes
:*

axis 
q
'decoder_1/layer_0/strided_slice_3/stackConst*
_output_shapes
:*
valueB: *
dtype0
s
)decoder_1/layer_0/strided_slice_3/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
s
)decoder_1/layer_0/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
!decoder_1/layer_0/strided_slice_3StridedSlicedecoder_1/layer_0/stack'decoder_1/layer_0/strided_slice_3/stack)decoder_1/layer_0/strided_slice_3/stack_1)decoder_1/layer_0/strided_slice_3/stack_2*
new_axis_mask *
shrink_axis_mask*
T0*
Index0*
ellipsis_mask *

begin_mask *
end_mask *
_output_shapes
: 
?
"decoder_1/layer_0/conv2d_transposeConv2DBackpropInputdecoder_1/layer_0/stackdecoder/layer_0/kernel/readdecoder_1/Reshape*
data_formatNHWC*
explicit_paddings
 *
strides
*
paddingVALID*0
_output_shapes
:??????????*
T0*
	dilations
*
use_cudnn_on_gpu(
?
decoder_1/layer_0/BiasAddBiasAdd"decoder_1/layer_0/conv2d_transposedecoder/layer_0/bias/read*
data_formatNHWC*
T0*0
_output_shapes
:??????????
f
#decoder_1/batch_normalization/ConstConst*
dtype0*
valueB *
_output_shapes
: 
h
%decoder_1/batch_normalization/Const_1Const*
dtype0*
valueB *
_output_shapes
: 
?
,decoder_1/batch_normalization/FusedBatchNormFusedBatchNormdecoder_1/layer_0/BiasAdd&decoder/batch_normalization/gamma/read%decoder/batch_normalization/beta/read#decoder_1/batch_normalization/Const%decoder_1/batch_normalization/Const_1*L
_output_shapes:
8:??????????:?:?:?:?*
epsilon%o?:*
T0*
is_training(*
data_formatNHWC
j
%decoder_1/batch_normalization/Const_2Const*
valueB
 *?p}?*
_output_shapes
: *
dtype0
?
3decoder_1/batch_normalization/AssignMovingAvg/sub/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0*:
_class0
.,loc:@decoder/batch_normalization/moving_mean
?
1decoder_1/batch_normalization/AssignMovingAvg/subSub3decoder_1/batch_normalization/AssignMovingAvg/sub/x%decoder_1/batch_normalization/Const_2*:
_class0
.,loc:@decoder/batch_normalization/moving_mean*
_output_shapes
: *
T0
?
3decoder_1/batch_normalization/AssignMovingAvg/sub_1Sub,decoder/batch_normalization/moving_mean/read.decoder_1/batch_normalization/FusedBatchNorm:1*:
_class0
.,loc:@decoder/batch_normalization/moving_mean*
T0*
_output_shapes	
:?
?
1decoder_1/batch_normalization/AssignMovingAvg/mulMul3decoder_1/batch_normalization/AssignMovingAvg/sub_11decoder_1/batch_normalization/AssignMovingAvg/sub*
T0*:
_class0
.,loc:@decoder/batch_normalization/moving_mean*
_output_shapes	
:?
?
-decoder_1/batch_normalization/AssignMovingAvg	AssignSub'decoder/batch_normalization/moving_mean1decoder_1/batch_normalization/AssignMovingAvg/mul*
use_locking( *:
_class0
.,loc:@decoder/batch_normalization/moving_mean*
_output_shapes	
:?*
T0
?
5decoder_1/batch_normalization/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *>
_class4
20loc:@decoder/batch_normalization/moving_variance*
valueB
 *  ??*
dtype0
?
3decoder_1/batch_normalization/AssignMovingAvg_1/subSub5decoder_1/batch_normalization/AssignMovingAvg_1/sub/x%decoder_1/batch_normalization/Const_2*
_output_shapes
: *>
_class4
20loc:@decoder/batch_normalization/moving_variance*
T0
?
5decoder_1/batch_normalization/AssignMovingAvg_1/sub_1Sub0decoder/batch_normalization/moving_variance/read.decoder_1/batch_normalization/FusedBatchNorm:2*
_output_shapes	
:?*>
_class4
20loc:@decoder/batch_normalization/moving_variance*
T0
?
3decoder_1/batch_normalization/AssignMovingAvg_1/mulMul5decoder_1/batch_normalization/AssignMovingAvg_1/sub_13decoder_1/batch_normalization/AssignMovingAvg_1/sub*
T0*
_output_shapes	
:?*>
_class4
20loc:@decoder/batch_normalization/moving_variance
?
/decoder_1/batch_normalization/AssignMovingAvg_1	AssignSub+decoder/batch_normalization/moving_variance3decoder_1/batch_normalization/AssignMovingAvg_1/mul*
_output_shapes	
:?*>
_class4
20loc:@decoder/batch_normalization/moving_variance*
use_locking( *
T0
?
decoder_1/LeakyRelu	LeakyRelu,decoder_1/batch_normalization/FusedBatchNorm*0
_output_shapes
:??????????*
T0*
alpha%??L>
j
decoder_1/layer_1/ShapeShapedecoder_1/LeakyRelu*
_output_shapes
:*
out_type0*
T0
o
%decoder_1/layer_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
q
'decoder_1/layer_1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
q
'decoder_1/layer_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
decoder_1/layer_1/strided_sliceStridedSlicedecoder_1/layer_1/Shape%decoder_1/layer_1/strided_slice/stack'decoder_1/layer_1/strided_slice/stack_1'decoder_1/layer_1/strided_slice/stack_2*

begin_mask *
Index0*
end_mask *
new_axis_mask *
_output_shapes
: *
T0*
shrink_axis_mask*
ellipsis_mask 
q
'decoder_1/layer_1/strided_slice_1/stackConst*
valueB:*
_output_shapes
:*
dtype0
s
)decoder_1/layer_1/strided_slice_1/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
s
)decoder_1/layer_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
!decoder_1/layer_1/strided_slice_1StridedSlicedecoder_1/layer_1/Shape'decoder_1/layer_1/strided_slice_1/stack)decoder_1/layer_1/strided_slice_1/stack_1)decoder_1/layer_1/strided_slice_1/stack_2*
new_axis_mask *
Index0*
end_mask *
_output_shapes
: *

begin_mask *
ellipsis_mask *
T0*
shrink_axis_mask
q
'decoder_1/layer_1/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
s
)decoder_1/layer_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
s
)decoder_1/layer_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
!decoder_1/layer_1/strided_slice_2StridedSlicedecoder_1/layer_1/Shape'decoder_1/layer_1/strided_slice_2/stack)decoder_1/layer_1/strided_slice_2/stack_1)decoder_1/layer_1/strided_slice_2/stack_2*
T0*
_output_shapes
: *

begin_mask *
shrink_axis_mask*
end_mask *
Index0*
ellipsis_mask *
new_axis_mask 
Y
decoder_1/layer_1/mul/yConst*
_output_shapes
: *
value	B :*
dtype0
y
decoder_1/layer_1/mulMul!decoder_1/layer_1/strided_slice_1decoder_1/layer_1/mul/y*
T0*
_output_shapes
: 
[
decoder_1/layer_1/mul_1/yConst*
dtype0*
value	B :*
_output_shapes
: 
}
decoder_1/layer_1/mul_1Mul!decoder_1/layer_1/strided_slice_2decoder_1/layer_1/mul_1/y*
T0*
_output_shapes
: 
\
decoder_1/layer_1/stack/3Const*
value
B :?*
dtype0*
_output_shapes
: 
?
decoder_1/layer_1/stackPackdecoder_1/layer_1/strided_slicedecoder_1/layer_1/muldecoder_1/layer_1/mul_1decoder_1/layer_1/stack/3*
N*
T0*
_output_shapes
:*

axis 
q
'decoder_1/layer_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 
s
)decoder_1/layer_1/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)decoder_1/layer_1/strided_slice_3/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
?
!decoder_1/layer_1/strided_slice_3StridedSlicedecoder_1/layer_1/stack'decoder_1/layer_1/strided_slice_3/stack)decoder_1/layer_1/strided_slice_3/stack_1)decoder_1/layer_1/strided_slice_3/stack_2*
_output_shapes
: *
Index0*
end_mask *
new_axis_mask *
shrink_axis_mask*

begin_mask *
ellipsis_mask *
T0
?
"decoder_1/layer_1/conv2d_transposeConv2DBackpropInputdecoder_1/layer_1/stackdecoder/layer_1/kernel/readdecoder_1/LeakyRelu*
explicit_paddings
 *
data_formatNHWC*
use_cudnn_on_gpu(*
strides
*0
_output_shapes
:??????????*
paddingSAME*
T0*
	dilations

?
decoder_1/layer_1/BiasAddBiasAdd"decoder_1/layer_1/conv2d_transposedecoder/layer_1/bias/read*
data_formatNHWC*
T0*0
_output_shapes
:??????????
h
%decoder_1/batch_normalization_1/ConstConst*
dtype0*
valueB *
_output_shapes
: 
j
'decoder_1/batch_normalization_1/Const_1Const*
dtype0*
_output_shapes
: *
valueB 
?
.decoder_1/batch_normalization_1/FusedBatchNormFusedBatchNormdecoder_1/layer_1/BiasAdd(decoder/batch_normalization_1/gamma/read'decoder/batch_normalization_1/beta/read%decoder_1/batch_normalization_1/Const'decoder_1/batch_normalization_1/Const_1*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
epsilon%o?:*
is_training(*
T0
l
'decoder_1/batch_normalization_1/Const_2Const*
_output_shapes
: *
valueB
 *?p}?*
dtype0
?
5decoder_1/batch_normalization_1/AssignMovingAvg/sub/xConst*
valueB
 *  ??*
dtype0*<
_class2
0.loc:@decoder/batch_normalization_1/moving_mean*
_output_shapes
: 
?
3decoder_1/batch_normalization_1/AssignMovingAvg/subSub5decoder_1/batch_normalization_1/AssignMovingAvg/sub/x'decoder_1/batch_normalization_1/Const_2*<
_class2
0.loc:@decoder/batch_normalization_1/moving_mean*
T0*
_output_shapes
: 
?
5decoder_1/batch_normalization_1/AssignMovingAvg/sub_1Sub.decoder/batch_normalization_1/moving_mean/read0decoder_1/batch_normalization_1/FusedBatchNorm:1*
T0*<
_class2
0.loc:@decoder/batch_normalization_1/moving_mean*
_output_shapes	
:?
?
3decoder_1/batch_normalization_1/AssignMovingAvg/mulMul5decoder_1/batch_normalization_1/AssignMovingAvg/sub_13decoder_1/batch_normalization_1/AssignMovingAvg/sub*
T0*
_output_shapes	
:?*<
_class2
0.loc:@decoder/batch_normalization_1/moving_mean
?
/decoder_1/batch_normalization_1/AssignMovingAvg	AssignSub)decoder/batch_normalization_1/moving_mean3decoder_1/batch_normalization_1/AssignMovingAvg/mul*
use_locking( *
T0*<
_class2
0.loc:@decoder/batch_normalization_1/moving_mean*
_output_shapes	
:?
?
7decoder_1/batch_normalization_1/AssignMovingAvg_1/sub/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0*@
_class6
42loc:@decoder/batch_normalization_1/moving_variance
?
5decoder_1/batch_normalization_1/AssignMovingAvg_1/subSub7decoder_1/batch_normalization_1/AssignMovingAvg_1/sub/x'decoder_1/batch_normalization_1/Const_2*
T0*@
_class6
42loc:@decoder/batch_normalization_1/moving_variance*
_output_shapes
: 
?
7decoder_1/batch_normalization_1/AssignMovingAvg_1/sub_1Sub2decoder/batch_normalization_1/moving_variance/read0decoder_1/batch_normalization_1/FusedBatchNorm:2*@
_class6
42loc:@decoder/batch_normalization_1/moving_variance*
_output_shapes	
:?*
T0
?
5decoder_1/batch_normalization_1/AssignMovingAvg_1/mulMul7decoder_1/batch_normalization_1/AssignMovingAvg_1/sub_15decoder_1/batch_normalization_1/AssignMovingAvg_1/sub*
T0*
_output_shapes	
:?*@
_class6
42loc:@decoder/batch_normalization_1/moving_variance
?
1decoder_1/batch_normalization_1/AssignMovingAvg_1	AssignSub-decoder/batch_normalization_1/moving_variance5decoder_1/batch_normalization_1/AssignMovingAvg_1/mul*
T0*@
_class6
42loc:@decoder/batch_normalization_1/moving_variance*
use_locking( *
_output_shapes	
:?
?
decoder_1/LeakyRelu_1	LeakyRelu.decoder_1/batch_normalization_1/FusedBatchNorm*0
_output_shapes
:??????????*
alpha%??L>*
T0
l
decoder_1/layer_2/ShapeShapedecoder_1/LeakyRelu_1*
out_type0*
_output_shapes
:*
T0
o
%decoder_1/layer_2/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
q
'decoder_1/layer_2/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
q
'decoder_1/layer_2/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
decoder_1/layer_2/strided_sliceStridedSlicedecoder_1/layer_2/Shape%decoder_1/layer_2/strided_slice/stack'decoder_1/layer_2/strided_slice/stack_1'decoder_1/layer_2/strided_slice/stack_2*
T0*

begin_mask *
new_axis_mask *
Index0*
ellipsis_mask *
shrink_axis_mask*
_output_shapes
: *
end_mask 
q
'decoder_1/layer_2/strided_slice_1/stackConst*
valueB:*
_output_shapes
:*
dtype0
s
)decoder_1/layer_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
s
)decoder_1/layer_2/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
!decoder_1/layer_2/strided_slice_1StridedSlicedecoder_1/layer_2/Shape'decoder_1/layer_2/strided_slice_1/stack)decoder_1/layer_2/strided_slice_1/stack_1)decoder_1/layer_2/strided_slice_1/stack_2*
Index0*

begin_mask *
_output_shapes
: *
ellipsis_mask *
T0*
new_axis_mask *
end_mask *
shrink_axis_mask
q
'decoder_1/layer_2/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
s
)decoder_1/layer_2/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)decoder_1/layer_2/strided_slice_2/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
!decoder_1/layer_2/strided_slice_2StridedSlicedecoder_1/layer_2/Shape'decoder_1/layer_2/strided_slice_2/stack)decoder_1/layer_2/strided_slice_2/stack_1)decoder_1/layer_2/strided_slice_2/stack_2*
T0*

begin_mask *
Index0*
_output_shapes
: *
ellipsis_mask *
new_axis_mask *
end_mask *
shrink_axis_mask
Y
decoder_1/layer_2/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
y
decoder_1/layer_2/mulMul!decoder_1/layer_2/strided_slice_1decoder_1/layer_2/mul/y*
T0*
_output_shapes
: 
[
decoder_1/layer_2/mul_1/yConst*
dtype0*
_output_shapes
: *
value	B :
}
decoder_1/layer_2/mul_1Mul!decoder_1/layer_2/strided_slice_2decoder_1/layer_2/mul_1/y*
T0*
_output_shapes
: 
[
decoder_1/layer_2/stack/3Const*
value	B :*
dtype0*
_output_shapes
: 
?
decoder_1/layer_2/stackPackdecoder_1/layer_2/strided_slicedecoder_1/layer_2/muldecoder_1/layer_2/mul_1decoder_1/layer_2/stack/3*
N*

axis *
T0*
_output_shapes
:
q
'decoder_1/layer_2/strided_slice_3/stackConst*
valueB: *
_output_shapes
:*
dtype0
s
)decoder_1/layer_2/strided_slice_3/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
s
)decoder_1/layer_2/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
!decoder_1/layer_2/strided_slice_3StridedSlicedecoder_1/layer_2/stack'decoder_1/layer_2/strided_slice_3/stack)decoder_1/layer_2/strided_slice_3/stack_1)decoder_1/layer_2/strided_slice_3/stack_2*
end_mask *
_output_shapes
: *
shrink_axis_mask*
new_axis_mask *
ellipsis_mask *

begin_mask *
Index0*
T0
?
"decoder_1/layer_2/conv2d_transposeConv2DBackpropInputdecoder_1/layer_2/stackdecoder/layer_2/kernel/readdecoder_1/LeakyRelu_1*
	dilations
*/
_output_shapes
:?????????*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
explicit_paddings
 *
T0
?
decoder_1/layer_2/BiasAddBiasAdd"decoder_1/layer_2/conv2d_transposedecoder/layer_2/bias/read*
data_formatNHWC*/
_output_shapes
:?????????*
T0
s
decoder_1/layer_2/TanhTanhdecoder_1/layer_2/BiasAdd*
T0*/
_output_shapes
:?????????
x
decoder_1/generated_imagesIdentitydecoder_1/layer_2/Tanh*
T0*/
_output_shapes
:?????????
l
disc0/Reshape/shapeConst*%
valueB"????         *
_output_shapes
:*
dtype0
x
disc0/ReshapeReshapeXdisc0/Reshape/shape*
T0*
Tshape0*/
_output_shapes
:?????????
?
6disc0/conv2d/kernel/Initializer/truncated_normal/shapeConst*&
_class
loc:@disc0/conv2d/kernel*%
valueB"            *
dtype0*
_output_shapes
:
?
5disc0/conv2d/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *
dtype0*&
_class
loc:@disc0/conv2d/kernel*
valueB
 *    
?
7disc0/conv2d/kernel/Initializer/truncated_normal/stddevConst*&
_class
loc:@disc0/conv2d/kernel*
dtype0*
valueB
 *??h>*
_output_shapes
: 
?
@disc0/conv2d/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal6disc0/conv2d/kernel/Initializer/truncated_normal/shape*
T0*
dtype0*

seed *
seed2 *&
_output_shapes
:*&
_class
loc:@disc0/conv2d/kernel
?
4disc0/conv2d/kernel/Initializer/truncated_normal/mulMul@disc0/conv2d/kernel/Initializer/truncated_normal/TruncatedNormal7disc0/conv2d/kernel/Initializer/truncated_normal/stddev*&
_output_shapes
:*&
_class
loc:@disc0/conv2d/kernel*
T0
?
0disc0/conv2d/kernel/Initializer/truncated_normalAdd4disc0/conv2d/kernel/Initializer/truncated_normal/mul5disc0/conv2d/kernel/Initializer/truncated_normal/mean*
T0*&
_output_shapes
:*&
_class
loc:@disc0/conv2d/kernel
?
disc0/conv2d/kernel
VariableV2*
shape:*
shared_name *&
_output_shapes
:*&
_class
loc:@disc0/conv2d/kernel*
dtype0*
	container 
?
disc0/conv2d/kernel/AssignAssigndisc0/conv2d/kernel0disc0/conv2d/kernel/Initializer/truncated_normal*&
_class
loc:@disc0/conv2d/kernel*
T0*
use_locking(*
validate_shape(*&
_output_shapes
:
?
disc0/conv2d/kernel/readIdentitydisc0/conv2d/kernel*
T0*&
_class
loc:@disc0/conv2d/kernel*&
_output_shapes
:
y
4disc0/conv2d/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *
valueB
 *o?:
z
5disc0/conv2d/kernel/Regularizer/l2_regularizer/L2LossL2Lossdisc0/conv2d/kernel/read*
_output_shapes
: *
T0
?
.disc0/conv2d/kernel/Regularizer/l2_regularizerMul4disc0/conv2d/kernel/Regularizer/l2_regularizer/scale5disc0/conv2d/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
#disc0/conv2d/bias/Initializer/zerosConst*$
_class
loc:@disc0/conv2d/bias*
dtype0*
_output_shapes
:*
valueB*    
?
disc0/conv2d/bias
VariableV2*
shared_name *
shape:*
	container *$
_class
loc:@disc0/conv2d/bias*
dtype0*
_output_shapes
:
?
disc0/conv2d/bias/AssignAssigndisc0/conv2d/bias#disc0/conv2d/bias/Initializer/zeros*$
_class
loc:@disc0/conv2d/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
?
disc0/conv2d/bias/readIdentitydisc0/conv2d/bias*
T0*
_output_shapes
:*$
_class
loc:@disc0/conv2d/bias
k
disc0/conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
disc0/conv2d/Conv2DConv2Ddisc0/Reshapedisc0/conv2d/kernel/read*
use_cudnn_on_gpu(*
paddingVALID*
strides
*/
_output_shapes
:?????????*
	dilations
*
explicit_paddings
 *
data_formatNHWC*
T0
?
disc0/conv2d/BiasAddBiasAdddisc0/conv2d/Conv2Ddisc0/conv2d/bias/read*
data_formatNHWC*
T0*/
_output_shapes
:?????????
i
disc0/conv2d/ReluReludisc0/conv2d/BiasAdd*/
_output_shapes
:?????????*
T0
?
disc0/MaxPoolMaxPooldisc0/conv2d/Relu*/
_output_shapes
:?????????*
T0*
paddingSAME*
data_formatNHWC*
strides
*
ksize

?
8disc0/conv2d_1/kernel/Initializer/truncated_normal/shapeConst*
dtype0*(
_class
loc:@disc0/conv2d_1/kernel*%
valueB"         2   *
_output_shapes
:
?
7disc0/conv2d_1/kernel/Initializer/truncated_normal/meanConst*(
_class
loc:@disc0/conv2d_1/kernel*
_output_shapes
: *
valueB
 *    *
dtype0
?
9disc0/conv2d_1/kernel/Initializer/truncated_normal/stddevConst*
dtype0*(
_class
loc:@disc0/conv2d_1/kernel*
valueB
 *?P=*
_output_shapes
: 
?
Bdisc0/conv2d_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal8disc0/conv2d_1/kernel/Initializer/truncated_normal/shape*&
_output_shapes
:2*
dtype0*

seed *
T0*(
_class
loc:@disc0/conv2d_1/kernel*
seed2 
?
6disc0/conv2d_1/kernel/Initializer/truncated_normal/mulMulBdisc0/conv2d_1/kernel/Initializer/truncated_normal/TruncatedNormal9disc0/conv2d_1/kernel/Initializer/truncated_normal/stddev*&
_output_shapes
:2*
T0*(
_class
loc:@disc0/conv2d_1/kernel
?
2disc0/conv2d_1/kernel/Initializer/truncated_normalAdd6disc0/conv2d_1/kernel/Initializer/truncated_normal/mul7disc0/conv2d_1/kernel/Initializer/truncated_normal/mean*
T0*(
_class
loc:@disc0/conv2d_1/kernel*&
_output_shapes
:2
?
disc0/conv2d_1/kernel
VariableV2*
shape:2*
	container *
dtype0*(
_class
loc:@disc0/conv2d_1/kernel*
shared_name *&
_output_shapes
:2
?
disc0/conv2d_1/kernel/AssignAssigndisc0/conv2d_1/kernel2disc0/conv2d_1/kernel/Initializer/truncated_normal*&
_output_shapes
:2*
T0*(
_class
loc:@disc0/conv2d_1/kernel*
validate_shape(*
use_locking(
?
disc0/conv2d_1/kernel/readIdentitydisc0/conv2d_1/kernel*&
_output_shapes
:2*
T0*(
_class
loc:@disc0/conv2d_1/kernel
{
6disc0/conv2d_1/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *
valueB
 *o?:
~
7disc0/conv2d_1/kernel/Regularizer/l2_regularizer/L2LossL2Lossdisc0/conv2d_1/kernel/read*
T0*
_output_shapes
: 
?
0disc0/conv2d_1/kernel/Regularizer/l2_regularizerMul6disc0/conv2d_1/kernel/Regularizer/l2_regularizer/scale7disc0/conv2d_1/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
%disc0/conv2d_1/bias/Initializer/zerosConst*
valueB2*    *
_output_shapes
:2*&
_class
loc:@disc0/conv2d_1/bias*
dtype0
?
disc0/conv2d_1/bias
VariableV2*
shared_name *
_output_shapes
:2*&
_class
loc:@disc0/conv2d_1/bias*
dtype0*
	container *
shape:2
?
disc0/conv2d_1/bias/AssignAssigndisc0/conv2d_1/bias%disc0/conv2d_1/bias/Initializer/zeros*&
_class
loc:@disc0/conv2d_1/bias*
_output_shapes
:2*
T0*
validate_shape(*
use_locking(
?
disc0/conv2d_1/bias/readIdentitydisc0/conv2d_1/bias*
_output_shapes
:2*
T0*&
_class
loc:@disc0/conv2d_1/bias
m
disc0/conv2d_1/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
?
disc0/conv2d_1/Conv2DConv2Ddisc0/MaxPooldisc0/conv2d_1/kernel/read*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*
T0*/
_output_shapes
:?????????2*
strides
*
data_formatNHWC*
	dilations

?
disc0/conv2d_1/BiasAddBiasAdddisc0/conv2d_1/Conv2Ddisc0/conv2d_1/bias/read*
T0*/
_output_shapes
:?????????2*
data_formatNHWC
m
disc0/conv2d_1/ReluReludisc0/conv2d_1/BiasAdd*
T0*/
_output_shapes
:?????????2
?
disc0/MaxPool_1MaxPooldisc0/conv2d_1/Relu*
strides
*/
_output_shapes
:?????????2*
T0*
ksize
*
data_formatNHWC*
paddingSAME
b
disc0/flatten/ShapeShapedisc0/MaxPool_1*
_output_shapes
:*
out_type0*
T0
k
!disc0/flatten/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
m
#disc0/flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
m
#disc0/flatten/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
disc0/flatten/strided_sliceStridedSlicedisc0/flatten/Shape!disc0/flatten/strided_slice/stack#disc0/flatten/strided_slice/stack_1#disc0/flatten/strided_slice/stack_2*
shrink_axis_mask*
Index0*
end_mask *
ellipsis_mask *

begin_mask *
T0*
_output_shapes
: *
new_axis_mask 
h
disc0/flatten/Reshape/shape/1Const*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
disc0/flatten/Reshape/shapePackdisc0/flatten/strided_slicedisc0/flatten/Reshape/shape/1*
N*

axis *
_output_shapes
:*
T0
?
disc0/flatten/ReshapeReshapedisc0/MaxPool_1disc0/flatten/Reshape/shape*(
_output_shapes
:??????????*
Tshape0*
T0
?
5disc0/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"   ?  *%
_class
loc:@disc0/dense/kernel*
dtype0*
_output_shapes
:
?
4disc0/dense/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *%
_class
loc:@disc0/dense/kernel*
dtype0
?
6disc0/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *?$=*%
_class
loc:@disc0/dense/kernel*
_output_shapes
: *
dtype0
?
?disc0/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal5disc0/dense/kernel/Initializer/truncated_normal/shape* 
_output_shapes
:
??*
T0*
dtype0*
seed2 *

seed *%
_class
loc:@disc0/dense/kernel
?
3disc0/dense/kernel/Initializer/truncated_normal/mulMul?disc0/dense/kernel/Initializer/truncated_normal/TruncatedNormal6disc0/dense/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*%
_class
loc:@disc0/dense/kernel*
T0
?
/disc0/dense/kernel/Initializer/truncated_normalAdd3disc0/dense/kernel/Initializer/truncated_normal/mul4disc0/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*%
_class
loc:@disc0/dense/kernel
?
disc0/dense/kernel
VariableV2*
	container *
shared_name *%
_class
loc:@disc0/dense/kernel* 
_output_shapes
:
??*
dtype0*
shape:
??
?
disc0/dense/kernel/AssignAssigndisc0/dense/kernel/disc0/dense/kernel/Initializer/truncated_normal*
validate_shape(*%
_class
loc:@disc0/dense/kernel*
T0*
use_locking(* 
_output_shapes
:
??
?
disc0/dense/kernel/readIdentitydisc0/dense/kernel* 
_output_shapes
:
??*%
_class
loc:@disc0/dense/kernel*
T0
x
3disc0/dense/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
valueB
 *o?:*
_output_shapes
: 
x
4disc0/dense/kernel/Regularizer/l2_regularizer/L2LossL2Lossdisc0/dense/kernel/read*
_output_shapes
: *
T0
?
-disc0/dense/kernel/Regularizer/l2_regularizerMul3disc0/dense/kernel/Regularizer/l2_regularizer/scale4disc0/dense/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
"disc0/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*
valueB?*    *#
_class
loc:@disc0/dense/bias
?
disc0/dense/bias
VariableV2*
dtype0*
shape:?*#
_class
loc:@disc0/dense/bias*
_output_shapes	
:?*
shared_name *
	container 
?
disc0/dense/bias/AssignAssigndisc0/dense/bias"disc0/dense/bias/Initializer/zeros*#
_class
loc:@disc0/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?
~
disc0/dense/bias/readIdentitydisc0/dense/bias*
T0*#
_class
loc:@disc0/dense/bias*
_output_shapes	
:?
?
disc0/dense/MatMulMatMuldisc0/flatten/Reshapedisc0/dense/kernel/read*
T0*
transpose_a( *(
_output_shapes
:??????????*
transpose_b( 
?
disc0/dense/BiasAddBiasAdddisc0/dense/MatMuldisc0/dense/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:??????????
`
disc0/dense/ReluReludisc0/dense/BiasAdd*
T0*(
_output_shapes
:??????????
?
7disc0/dense_1/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"?     *
dtype0*'
_class
loc:@disc0/dense_1/kernel
?
6disc0/dense_1/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0*'
_class
loc:@disc0/dense_1/kernel
?
8disc0/dense_1/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *'
_class
loc:@disc0/dense_1/kernel*
dtype0*
valueB
 *?P=
?
Adisc0/dense_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal7disc0/dense_1/kernel/Initializer/truncated_normal/shape*'
_class
loc:@disc0/dense_1/kernel*
_output_shapes
:	?*
dtype0*
T0*
seed2 *

seed 
?
5disc0/dense_1/kernel/Initializer/truncated_normal/mulMulAdisc0/dense_1/kernel/Initializer/truncated_normal/TruncatedNormal8disc0/dense_1/kernel/Initializer/truncated_normal/stddev*
_output_shapes
:	?*'
_class
loc:@disc0/dense_1/kernel*
T0
?
1disc0/dense_1/kernel/Initializer/truncated_normalAdd5disc0/dense_1/kernel/Initializer/truncated_normal/mul6disc0/dense_1/kernel/Initializer/truncated_normal/mean*'
_class
loc:@disc0/dense_1/kernel*
_output_shapes
:	?*
T0
?
disc0/dense_1/kernel
VariableV2*
shape:	?*
shared_name *
	container *'
_class
loc:@disc0/dense_1/kernel*
_output_shapes
:	?*
dtype0
?
disc0/dense_1/kernel/AssignAssigndisc0/dense_1/kernel1disc0/dense_1/kernel/Initializer/truncated_normal*
use_locking(*'
_class
loc:@disc0/dense_1/kernel*
validate_shape(*
_output_shapes
:	?*
T0
?
disc0/dense_1/kernel/readIdentitydisc0/dense_1/kernel*
T0*
_output_shapes
:	?*'
_class
loc:@disc0/dense_1/kernel
z
5disc0/dense_1/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o?:*
dtype0*
_output_shapes
: 
|
6disc0/dense_1/kernel/Regularizer/l2_regularizer/L2LossL2Lossdisc0/dense_1/kernel/read*
_output_shapes
: *
T0
?
/disc0/dense_1/kernel/Regularizer/l2_regularizerMul5disc0/dense_1/kernel/Regularizer/l2_regularizer/scale6disc0/dense_1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
$disc0/dense_1/bias/Initializer/zerosConst*
valueB*    *%
_class
loc:@disc0/dense_1/bias*
dtype0*
_output_shapes
:
?
disc0/dense_1/bias
VariableV2*
	container *
dtype0*
_output_shapes
:*%
_class
loc:@disc0/dense_1/bias*
shared_name *
shape:
?
disc0/dense_1/bias/AssignAssigndisc0/dense_1/bias$disc0/dense_1/bias/Initializer/zeros*%
_class
loc:@disc0/dense_1/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
?
disc0/dense_1/bias/readIdentitydisc0/dense_1/bias*
T0*
_output_shapes
:*%
_class
loc:@disc0/dense_1/bias
?
disc0/dense_1/MatMulMatMuldisc0/dense/Reludisc0/dense_1/kernel/read*
T0*
transpose_b( *'
_output_shapes
:?????????*
transpose_a( 
?
disc0/dense_1/BiasAddBiasAdddisc0/dense_1/MatMuldisc0/dense_1/bias/read*'
_output_shapes
:?????????*
data_formatNHWC*
T0
]
	Sigmoid_2Sigmoiddisc0/dense_1/BiasAdd*'
_output_shapes
:?????????*
T0
n
disc0_1/Reshape/shapeConst*
dtype0*%
valueB"????         *
_output_shapes
:
?
disc0_1/ReshapeReshapedecoder_1/generated_imagesdisc0_1/Reshape/shape*
Tshape0*/
_output_shapes
:?????????*
T0
m
disc0_1/conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
disc0_1/conv2d/Conv2DConv2Ddisc0_1/Reshapedisc0/conv2d/kernel/read*
strides
*/
_output_shapes
:?????????*
use_cudnn_on_gpu(*
T0*
data_formatNHWC*
paddingVALID*
explicit_paddings
 *
	dilations

?
disc0_1/conv2d/BiasAddBiasAdddisc0_1/conv2d/Conv2Ddisc0/conv2d/bias/read*
data_formatNHWC*/
_output_shapes
:?????????*
T0
m
disc0_1/conv2d/ReluReludisc0_1/conv2d/BiasAdd*/
_output_shapes
:?????????*
T0
?
disc0_1/MaxPoolMaxPooldisc0_1/conv2d/Relu*
T0*
ksize
*
paddingSAME*
data_formatNHWC*/
_output_shapes
:?????????*
strides

o
disc0_1/conv2d_1/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
disc0_1/conv2d_1/Conv2DConv2Ddisc0_1/MaxPooldisc0/conv2d_1/kernel/read*
use_cudnn_on_gpu(*
data_formatNHWC*
	dilations
*
T0*
strides
*
paddingVALID*
explicit_paddings
 */
_output_shapes
:?????????2
?
disc0_1/conv2d_1/BiasAddBiasAdddisc0_1/conv2d_1/Conv2Ddisc0/conv2d_1/bias/read*/
_output_shapes
:?????????2*
data_formatNHWC*
T0
q
disc0_1/conv2d_1/ReluReludisc0_1/conv2d_1/BiasAdd*
T0*/
_output_shapes
:?????????2
?
disc0_1/MaxPool_1MaxPooldisc0_1/conv2d_1/Relu*
strides
*
paddingSAME*
T0*
data_formatNHWC*
ksize
*/
_output_shapes
:?????????2
f
disc0_1/flatten/ShapeShapedisc0_1/MaxPool_1*
_output_shapes
:*
out_type0*
T0
m
#disc0_1/flatten/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
o
%disc0_1/flatten/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
o
%disc0_1/flatten/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
disc0_1/flatten/strided_sliceStridedSlicedisc0_1/flatten/Shape#disc0_1/flatten/strided_slice/stack%disc0_1/flatten/strided_slice/stack_1%disc0_1/flatten/strided_slice/stack_2*
T0*

begin_mask *
shrink_axis_mask*
ellipsis_mask *
_output_shapes
: *
new_axis_mask *
Index0*
end_mask 
j
disc0_1/flatten/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????
?
disc0_1/flatten/Reshape/shapePackdisc0_1/flatten/strided_slicedisc0_1/flatten/Reshape/shape/1*
_output_shapes
:*
N*

axis *
T0
?
disc0_1/flatten/ReshapeReshapedisc0_1/MaxPool_1disc0_1/flatten/Reshape/shape*
T0*(
_output_shapes
:??????????*
Tshape0
?
disc0_1/dense/MatMulMatMuldisc0_1/flatten/Reshapedisc0/dense/kernel/read*(
_output_shapes
:??????????*
transpose_b( *
T0*
transpose_a( 
?
disc0_1/dense/BiasAddBiasAdddisc0_1/dense/MatMuldisc0/dense/bias/read*
T0*(
_output_shapes
:??????????*
data_formatNHWC
d
disc0_1/dense/ReluReludisc0_1/dense/BiasAdd*(
_output_shapes
:??????????*
T0
?
disc0_1/dense_1/MatMulMatMuldisc0_1/dense/Reludisc0/dense_1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:?????????*
transpose_a( 
?
disc0_1/dense_1/BiasAddBiasAdddisc0_1/dense_1/MatMuldisc0/dense_1/bias/read*
data_formatNHWC*'
_output_shapes
:?????????*
T0
_
	Sigmoid_3Sigmoiddisc0_1/dense_1/BiasAdd*'
_output_shapes
:?????????*
T0
a
zeros_1/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:d
R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
n
zeros_1Fillzeros_1/shape_as_tensorzeros_1/Const*
T0*
_output_shapes
:d*

index_type0
`
ones_1/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:d
Q
ones_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
k
ones_1Fillones_1/shape_as_tensorones_1/Const*
_output_shapes
:d*

index_type0*
T0
?
bMultivariateNormalDiag_1/shapes_from_loc_and_scale/LinearOperatorDiag/range_dimension_tensor/ConstConst*
value	B :d*
dtype0*
_output_shapes
: 
?
>MultivariateNormalDiag_1/shapes_from_loc_and_scale/event_shapeConst*
dtype0*
valueB:d*
_output_shapes
:
?
dMultivariateNormalDiag_1/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/batch_shapeConst*
_output_shapes
: *
dtype0*
valueB 
?
BMultivariateNormalDiag_1/shapes_from_loc_and_scale/loc_batch_shapeConst*
valueB *
_output_shapes
: *
dtype0
?
>MultivariateNormalDiag_1/shapes_from_loc_and_scale/batch_shapeConst*
_output_shapes
: *
valueB *
dtype0
L
zeros_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
K
ones_2Const*
dtype0*
valueB
 *  ??*
_output_shapes
: 
G
Normal_1/IdentityIdentityzeros_2*
_output_shapes
: *
T0
H
Normal_1/Identity_1Identityones_2*
_output_shapes
: *
T0
_
MultivariateNormalDiag_1/zeroConst*
_output_shapes
: *
value	B : *
dtype0
a
MultivariateNormalDiag_1/emptyConst*
dtype0*
valueB *
_output_shapes
: 
j
(Normal_1/is_scalar_batch/is_scalar_batchConst*
value	B
 Z*
_output_shapes
: *
dtype0

j
(Normal_1/is_scalar_event/is_scalar_eventConst*
_output_shapes
: *
dtype0
*
value	B
 Z
l
*Normal_1/is_scalar_batch_1/is_scalar_batchConst*
value	B
 Z*
_output_shapes
: *
dtype0

`
MultivariateNormalDiag_1/ConstConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
o
$MultivariateNormalDiag_1/range/startConst*
dtype0*
_output_shapes
: *
valueB :
?????????
f
$MultivariateNormalDiag_1/range/limitConst*
_output_shapes
: *
value	B : *
dtype0
f
$MultivariateNormalDiag_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
MultivariateNormalDiag_1/rangeRange$MultivariateNormalDiag_1/range/start$MultivariateNormalDiag_1/range/limit$MultivariateNormalDiag_1/range/delta*

Tidx0*
_output_shapes
:
?
DMultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/subSubHencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/addzeros_1*'
_output_shapes
:?????????d*
T0
?
hMultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
dMultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims
ExpandDimsDMultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/subhMultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims/dim*

Tdim0*+
_output_shapes
:?????????d*
T0
?
|MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
zMultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/truedivRealDiv|MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/truediv/xones_1*
T0*
_output_shapes
:d
?
?MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
?????????
?
}MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/ExpandDims
ExpandDimszMultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/truediv?MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/ExpandDims/dim*
T0*

Tdim0*
_output_shapes

:d
?
vMultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mulMuldMultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims}MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/ExpandDims*
T0*+
_output_shapes
:?????????d
?
aMultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/SqueezeSqueezevMultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul*
T0*
squeeze_dims

?????????*'
_output_shapes
:?????????d
?
_MultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/event_ndimsConst*
dtype0*
_output_shapes
: *
value	B :
?
vMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/AbsAbsones_1*
_output_shapes
:d*
T0
?
vMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/LogLogvMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Abs*
T0*
_output_shapes
:d
?
?MultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum/reduction_indicesConst*
dtype0*
valueB:
?????????*
_output_shapes
:
?
vMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/SumSumvMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Log?MultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
?
WMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/NegNegvMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/LinearOperatorDiag/log_abs_det/Sum*
T0*
_output_shapes
: 
?
YMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/ShapeShapeaMultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/Squeeze*
out_type0*
T0*
_output_shapes
:
?
XMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/SizeConst*
value	B :*
_output_shapes
: *
dtype0
?
YMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/sub/yConst*
_output_shapes
: *
value	B :*
dtype0
?
WMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/subSubXMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/SizeYMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/sub/y*
_output_shapes
: *
T0
?
[MultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :
?
YMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/sub_1SubXMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/Size[MultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/sub_1/y*
_output_shapes
: *
T0
?
gMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/strided_slice/stackPackWMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/sub*
N*
T0*

axis *
_output_shapes
:
?
iMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/strided_slice/stack_1PackYMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/sub_1*

axis *
T0*
N*
_output_shapes
:
?
iMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
aMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/strided_sliceStridedSliceYMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/ShapegMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/strided_slice/stackiMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/strided_slice/stack_1iMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/strided_slice/stack_2*
_output_shapes
: *

begin_mask *
end_mask *
Index0*
ellipsis_mask *
T0*
new_axis_mask *
shrink_axis_mask 
?
^MultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
XMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/onesFillaMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/strided_slice^MultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/ones/Const*
_output_shapes
: *

index_type0*
T0
?
WMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/mulMulXMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/onesWMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/Neg*
T0*
_output_shapes
: 
?
iMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/Sum/reduction_indicesConst*
_output_shapes
: *
valueB *
dtype0
?
WMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/SumSumWMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/muliMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/Sum/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
?
!Normal_1/log_prob/standardize/subSubaMultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/SqueezeNormal_1/Identity*'
_output_shapes
:?????????d*
T0
?
%Normal_1/log_prob/standardize/truedivRealDiv!Normal_1/log_prob/standardize/subNormal_1/Identity_1*'
_output_shapes
:?????????d*
T0
{
Normal_1/log_prob/SquareSquare%Normal_1/log_prob/standardize/truediv*'
_output_shapes
:?????????d*
T0
\
Normal_1/log_prob/mul/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
?
Normal_1/log_prob/mulMulNormal_1/log_prob/mul/xNormal_1/log_prob/Square*
T0*'
_output_shapes
:?????????d
R
Normal_1/log_prob/LogLogNormal_1/Identity_1*
_output_shapes
: *
T0
\
Normal_1/log_prob/add/xConst*
dtype0*
valueB
 *??k?*
_output_shapes
: 
m
Normal_1/log_prob/addAddNormal_1/log_prob/add/xNormal_1/log_prob/Log*
T0*
_output_shapes
: 
|
Normal_1/log_prob/subSubNormal_1/log_prob/mulNormal_1/log_prob/add*
T0*'
_output_shapes
:?????????d
?
%MultivariateNormalDiag_1/log_prob/SumSumNormal_1/log_prob/subMultivariateNormalDiag_1/range*#
_output_shapes
:?????????*
T0*

Tidx0*
	keep_dims( 
?
%MultivariateNormalDiag_1/log_prob/addAdd%MultivariateNormalDiag_1/log_prob/SumWMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/Sum*
T0*#
_output_shapes
:?????????
O
ConstConst*
_output_shapes
:*
dtype0*
valueB: 
x
MeanMean%MultivariateNormalDiag_1/log_prob/addConst*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
1
NegNegMean*
T0*
_output_shapes
: 
?
FMultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/subSub@MultivariateNormalDiag/sample/affine_linear_operator/forward/addzeros_1*'
_output_shapes
:?????????d*
T0
?
jMultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims/dimConst*
_output_shapes
: *
valueB :
?????????*
dtype0
?
fMultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims
ExpandDimsFMultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/subjMultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims/dim*

Tdim0*
T0*+
_output_shapes
:?????????d
?
~MultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/truediv/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
|MultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/truedivRealDiv~MultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/truediv/xones_1*
_output_shapes
:d*
T0
?
?MultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
?????????
?
MultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/ExpandDims
ExpandDims|MultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/truediv?MultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/ExpandDims/dim*
T0*

Tdim0*
_output_shapes

:d
?
xMultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mulMulfMultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDimsMultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/ExpandDims*
T0*+
_output_shapes
:?????????d
?
cMultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/LinearOperatorDiag/solve/SqueezeSqueezexMultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul*
squeeze_dims

?????????*
T0*'
_output_shapes
:?????????d
?
aMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/event_ndimsConst*
value	B :*
dtype0*
_output_shapes
: 
?
[MultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/ShapeShapecMultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/LinearOperatorDiag/solve/Squeeze*
_output_shapes
:*
T0*
out_type0
?
ZMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
?
[MultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
?
YMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/subSubZMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/Size[MultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/sub/y*
_output_shapes
: *
T0
?
]MultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/sub_1/yConst*
value	B :*
_output_shapes
: *
dtype0
?
[MultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/sub_1SubZMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/Size]MultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/sub_1/y*
T0*
_output_shapes
: 
?
iMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/strided_slice/stackPackYMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/sub*
T0*
N*

axis *
_output_shapes
:
?
kMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/strided_slice/stack_1Pack[MultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/sub_1*

axis *
_output_shapes
:*
T0*
N
?
kMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
cMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/strided_sliceStridedSlice[MultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/ShapeiMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/strided_slice/stackkMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/strided_slice/stack_1kMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/strided_slice/stack_2*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask *

begin_mask *
end_mask *
new_axis_mask *
ellipsis_mask 
?
`MultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
ZMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/onesFillcMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/strided_slice`MultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/ones/Const*
T0*
_output_shapes
: *

index_type0
?
YMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/mulMulZMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/onesWMultivariateNormalDiag_1/log_prob/affine_linear_operator_1/inverse_log_det_jacobian/Neg*
T0*
_output_shapes
: 
?
kMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/Sum/reduction_indicesConst*
dtype0*
valueB *
_output_shapes
: 
?
YMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/SumSumYMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/mulkMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
?
#Normal_1/log_prob_1/standardize/subSubcMultivariateNormalDiag_1/log_prob_1/affine_linear_operator/inverse/LinearOperatorDiag/solve/SqueezeNormal_1/Identity*
T0*'
_output_shapes
:?????????d
?
'Normal_1/log_prob_1/standardize/truedivRealDiv#Normal_1/log_prob_1/standardize/subNormal_1/Identity_1*
T0*'
_output_shapes
:?????????d

Normal_1/log_prob_1/SquareSquare'Normal_1/log_prob_1/standardize/truediv*'
_output_shapes
:?????????d*
T0
^
Normal_1/log_prob_1/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
?
Normal_1/log_prob_1/mulMulNormal_1/log_prob_1/mul/xNormal_1/log_prob_1/Square*
T0*'
_output_shapes
:?????????d
T
Normal_1/log_prob_1/LogLogNormal_1/Identity_1*
_output_shapes
: *
T0
^
Normal_1/log_prob_1/add/xConst*
dtype0*
valueB
 *??k?*
_output_shapes
: 
s
Normal_1/log_prob_1/addAddNormal_1/log_prob_1/add/xNormal_1/log_prob_1/Log*
_output_shapes
: *
T0
?
Normal_1/log_prob_1/subSubNormal_1/log_prob_1/mulNormal_1/log_prob_1/add*
T0*'
_output_shapes
:?????????d
?
'MultivariateNormalDiag_1/log_prob_1/SumSumNormal_1/log_prob_1/subMultivariateNormalDiag_1/range*#
_output_shapes
:?????????*
T0*
	keep_dims( *

Tidx0
?
'MultivariateNormalDiag_1/log_prob_1/addAdd'MultivariateNormalDiag_1/log_prob_1/SumYMultivariateNormalDiag_1/log_prob_1/affine_linear_operator_1/inverse_log_det_jacobian/Sum*
T0*#
_output_shapes
:?????????
Q
Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
~
Mean_1Mean'MultivariateNormalDiag_1/log_prob_1/addConst_1*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
5
Neg_1NegMean_1*
T0*
_output_shapes
: 
L
add_1/yConst*
valueB
 *w?+2*
dtype0*
_output_shapes
: 
P
add_1AddSigmoidadd_1/y*
T0*'
_output_shapes
:?????????
C
LogLogadd_1*
T0*'
_output_shapes
:?????????
C
Neg_2NegLog*'
_output_shapes
:?????????*
T0
X
Const_2Const*
dtype0*
_output_shapes
:*
valueB"       
\
Mean_2MeanNeg_2Const_2*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
J
sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
N
subSubsub/x	Sigmoid_1*
T0*'
_output_shapes
:?????????
L
add_2/yConst*
dtype0*
valueB
 *w?+2*
_output_shapes
: 
L
add_2Addsubadd_2/y*'
_output_shapes
:?????????*
T0
E
Log_1Logadd_2*'
_output_shapes
:?????????*
T0
E
Neg_3NegLog_1*
T0*'
_output_shapes
:?????????
X
Const_3Const*
valueB"       *
dtype0*
_output_shapes
:
\
Mean_3MeanNeg_3Const_3*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
=
add_3AddMean_2Mean_3*
_output_shapes
: *
T0
L
add_4/yConst*
dtype0*
_output_shapes
: *
valueB
 *w?+2
R
add_4Add	Sigmoid_2add_4/y*
T0*'
_output_shapes
:?????????
E
Log_2Logadd_4*
T0*'
_output_shapes
:?????????
E
Neg_4NegLog_2*'
_output_shapes
:?????????*
T0
X
Const_4Const*
_output_shapes
:*
valueB"       *
dtype0
\
Mean_4MeanNeg_4Const_4*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
L
sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
R
sub_1Subsub_1/x	Sigmoid_3*
T0*'
_output_shapes
:?????????
L
add_5/yConst*
valueB
 *w?+2*
_output_shapes
: *
dtype0
N
add_5Addsub_1add_5/y*'
_output_shapes
:?????????*
T0
E
Log_3Logadd_5*'
_output_shapes
:?????????*
T0
E
Neg_5NegLog_3*
T0*'
_output_shapes
:?????????
X
Const_5Const*
dtype0*
_output_shapes
:*
valueB"       
\
Mean_5MeanNeg_5Const_5*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
=
add_6AddMean_4Mean_5*
_output_shapes
: *
T0
L
sub_2/xConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
R
sub_2Subsub_2/x	Sigmoid_3*'
_output_shapes
:?????????*
T0
L
add_7/yConst*
valueB
 *w?+2*
dtype0*
_output_shapes
: 
N
add_7Addsub_2add_7/y*
T0*'
_output_shapes
:?????????
E
Log_4Logadd_7*'
_output_shapes
:?????????*
T0
E
Neg_6NegLog_4*'
_output_shapes
:?????????*
T0
X
Const_6Const*
dtype0*
_output_shapes
:*
valueB"       
\
Mean_6MeanNeg_6Const_6*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
L
add_8/yConst*
dtype0*
_output_shapes
: *
valueB
 *w?+2
R
add_8Add	Sigmoid_1add_8/y*'
_output_shapes
:?????????*
T0
E
Log_5Logadd_8*
T0*'
_output_shapes
:?????????
E
Neg_7NegLog_5*'
_output_shapes
:?????????*
T0
X
Const_7Const*
_output_shapes
:*
dtype0*
valueB"       
\
Mean_7MeanNeg_7Const_7*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
=
add_9AddMean_6Mean_7*
T0*
_output_shapes
: 
e
flatten/ShapeShapedecoder/generated_images*
_output_shapes
:*
out_type0*
T0
e
flatten/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
g
flatten/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
g
flatten/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
flatten/strided_sliceStridedSliceflatten/Shapeflatten/strided_slice/stackflatten/strided_slice/stack_1flatten/strided_slice/stack_2*
end_mask *
_output_shapes
: *
shrink_axis_mask*
ellipsis_mask *
new_axis_mask *
Index0*

begin_mask *
T0
b
flatten/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????
?
flatten/Reshape/shapePackflatten/strided_sliceflatten/Reshape/shape/1*
_output_shapes
:*
T0*

axis *
N
?
flatten/ReshapeReshapedecoder/generated_imagesflatten/Reshape/shape*
Tshape0*(
_output_shapes
:??????????*
T0
S
sub_3Subflatten/ReshapeX*(
_output_shapes
:??????????*
T0
J
SquareSquaresub_3*
T0*(
_output_shapes
:??????????
W
Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
t
SumSumSquareSum/reduction_indices*
T0*#
_output_shapes
:?????????*

Tidx0*
	keep_dims( 
M
add_10/yConst*
dtype0*
_output_shapes
: *
valueB
 *??8
J
add_10AddSumadd_10/y*#
_output_shapes
:?????????*
T0
Q
Const_8Const*
valueB: *
dtype0*
_output_shapes
:
]
Mean_8Meanadd_10Const_8*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=
7
mulMulmul/xNeg*
T0*
_output_shapes
: 
;
add_11AddMean_8mul*
T0*
_output_shapes
: 
`

zeros_like	ZerosLikediz_1/dense_3/BiasAdd*'
_output_shapes
:?????????*
T0
n
logistic_loss/zeros_like	ZerosLikediz_1/dense_3/BiasAdd*
T0*'
_output_shapes
:?????????
?
logistic_loss/GreaterEqualGreaterEqualdiz_1/dense_3/BiasAddlogistic_loss/zeros_like*'
_output_shapes
:?????????*
T0
?
logistic_loss/SelectSelectlogistic_loss/GreaterEqualdiz_1/dense_3/BiasAddlogistic_loss/zeros_like*'
_output_shapes
:?????????*
T0
a
logistic_loss/NegNegdiz_1/dense_3/BiasAdd*'
_output_shapes
:?????????*
T0
?
logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/Negdiz_1/dense_3/BiasAdd*
T0*'
_output_shapes
:?????????
m
logistic_loss/mulMuldiz_1/dense_3/BiasAdd
zeros_like*'
_output_shapes
:?????????*
T0
s
logistic_loss/subSublogistic_loss/Selectlogistic_loss/mul*'
_output_shapes
:?????????*
T0
b
logistic_loss/ExpExplogistic_loss/Select_1*
T0*'
_output_shapes
:?????????
a
logistic_loss/Log1pLog1plogistic_loss/Exp*'
_output_shapes
:?????????*
T0
n
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*'
_output_shapes
:?????????*
T0
X
Const_9Const*
dtype0*
_output_shapes
:*
valueB"       
d
Mean_9Meanlogistic_lossConst_9*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
>
add_12Addadd_11Mean_9*
_output_shapes
: *
T0
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
X
gradients/grad_ys_0Const*
valueB
 *  ??*
_output_shapes
: *
dtype0
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
?
&gradients/add_12_grad/tuple/group_depsNoOp^gradients/Fill
?
.gradients/add_12_grad/tuple/control_dependencyIdentitygradients/Fill'^gradients/add_12_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
_output_shapes
: *
T0
?
0gradients/add_12_grad/tuple/control_dependency_1Identitygradients/Fill'^gradients/add_12_grad/tuple/group_deps*
_output_shapes
: *!
_class
loc:@gradients/Fill*
T0
_
&gradients/add_11_grad/tuple/group_depsNoOp/^gradients/add_12_grad/tuple/control_dependency
?
.gradients/add_11_grad/tuple/control_dependencyIdentity.gradients/add_12_grad/tuple/control_dependency'^gradients/add_11_grad/tuple/group_deps*
_output_shapes
: *
T0*!
_class
loc:@gradients/Fill
?
0gradients/add_11_grad/tuple/control_dependency_1Identity.gradients/add_12_grad/tuple/control_dependency'^gradients/add_11_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
_output_shapes
: *
T0
t
#gradients/Mean_9_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
?
gradients/Mean_9_grad/ReshapeReshape0gradients/add_12_grad/tuple/control_dependency_1#gradients/Mean_9_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes

:
h
gradients/Mean_9_grad/ShapeShapelogistic_loss*
_output_shapes
:*
T0*
out_type0
?
gradients/Mean_9_grad/TileTilegradients/Mean_9_grad/Reshapegradients/Mean_9_grad/Shape*
T0*'
_output_shapes
:?????????*

Tmultiples0
j
gradients/Mean_9_grad/Shape_1Shapelogistic_loss*
T0*
out_type0*
_output_shapes
:
`
gradients/Mean_9_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
e
gradients/Mean_9_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
gradients/Mean_9_grad/ProdProdgradients/Mean_9_grad/Shape_1gradients/Mean_9_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
g
gradients/Mean_9_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
?
gradients/Mean_9_grad/Prod_1Prodgradients/Mean_9_grad/Shape_2gradients/Mean_9_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
a
gradients/Mean_9_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
?
gradients/Mean_9_grad/MaximumMaximumgradients/Mean_9_grad/Prod_1gradients/Mean_9_grad/Maximum/y*
_output_shapes
: *
T0
?
gradients/Mean_9_grad/floordivFloorDivgradients/Mean_9_grad/Prodgradients/Mean_9_grad/Maximum*
_output_shapes
: *
T0
?
gradients/Mean_9_grad/CastCastgradients/Mean_9_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0*
Truncate( 
?
gradients/Mean_9_grad/truedivRealDivgradients/Mean_9_grad/Tilegradients/Mean_9_grad/Cast*'
_output_shapes
:?????????*
T0
m
#gradients/Mean_8_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?
gradients/Mean_8_grad/ReshapeReshape.gradients/add_11_grad/tuple/control_dependency#gradients/Mean_8_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
a
gradients/Mean_8_grad/ShapeShapeadd_10*
T0*
_output_shapes
:*
out_type0
?
gradients/Mean_8_grad/TileTilegradients/Mean_8_grad/Reshapegradients/Mean_8_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:?????????
c
gradients/Mean_8_grad/Shape_1Shapeadd_10*
out_type0*
_output_shapes
:*
T0
`
gradients/Mean_8_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
e
gradients/Mean_8_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
?
gradients/Mean_8_grad/ProdProdgradients/Mean_8_grad/Shape_1gradients/Mean_8_grad/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
g
gradients/Mean_8_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
?
gradients/Mean_8_grad/Prod_1Prodgradients/Mean_8_grad/Shape_2gradients/Mean_8_grad/Const_1*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0
a
gradients/Mean_8_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
?
gradients/Mean_8_grad/MaximumMaximumgradients/Mean_8_grad/Prod_1gradients/Mean_8_grad/Maximum/y*
T0*
_output_shapes
: 
?
gradients/Mean_8_grad/floordivFloorDivgradients/Mean_8_grad/Prodgradients/Mean_8_grad/Maximum*
T0*
_output_shapes
: 
?
gradients/Mean_8_grad/CastCastgradients/Mean_8_grad/floordiv*

DstT0*
_output_shapes
: *

SrcT0*
Truncate( 
?
gradients/Mean_8_grad/truedivRealDivgradients/Mean_8_grad/Tilegradients/Mean_8_grad/Cast*
T0*#
_output_shapes
:?????????
u
gradients/mul_grad/MulMul0gradients/add_11_grad/tuple/control_dependency_1Neg*
_output_shapes
: *
T0
y
gradients/mul_grad/Mul_1Mul0gradients/add_11_grad/tuple/control_dependency_1mul/x*
_output_shapes
: *
T0
_
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Mul^gradients/mul_grad/Mul_1
?
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Mul$^gradients/mul_grad/tuple/group_deps*
_output_shapes
: *)
_class
loc:@gradients/mul_grad/Mul*
T0
?
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Mul_1$^gradients/mul_grad/tuple/group_deps*+
_class!
loc:@gradients/mul_grad/Mul_1*
T0*
_output_shapes
: 
s
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub*
_output_shapes
:*
T0*
out_type0
w
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
_output_shapes
:*
out_type0*
T0
?
2gradients/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/logistic_loss_grad/Shape$gradients/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
 gradients/logistic_loss_grad/SumSumgradients/Mean_9_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape*'
_output_shapes
:?????????*
T0*
Tshape0
?
"gradients/logistic_loss_grad/Sum_1Sumgradients/Mean_9_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
?
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
-gradients/logistic_loss_grad/tuple/group_depsNoOp%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1
?
5gradients/logistic_loss_grad/tuple/control_dependencyIdentity$gradients/logistic_loss_grad/Reshape.^gradients/logistic_loss_grad/tuple/group_deps*'
_output_shapes
:?????????*7
_class-
+)loc:@gradients/logistic_loss_grad/Reshape*
T0
?
7gradients/logistic_loss_grad/tuple/control_dependency_1Identity&gradients/logistic_loss_grad/Reshape_1.^gradients/logistic_loss_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/logistic_loss_grad/Reshape_1*'
_output_shapes
:?????????
^
gradients/add_10_grad/ShapeShapeSum*
out_type0*
_output_shapes
:*
T0
`
gradients/add_10_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
?
+gradients/add_10_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_10_grad/Shapegradients/add_10_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/add_10_grad/SumSumgradients/Mean_8_grad/truediv+gradients/add_10_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
?
gradients/add_10_grad/ReshapeReshapegradients/add_10_grad/Sumgradients/add_10_grad/Shape*
T0*#
_output_shapes
:?????????*
Tshape0
?
gradients/add_10_grad/Sum_1Sumgradients/Mean_8_grad/truediv-gradients/add_10_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
?
gradients/add_10_grad/Reshape_1Reshapegradients/add_10_grad/Sum_1gradients/add_10_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
p
&gradients/add_10_grad/tuple/group_depsNoOp^gradients/add_10_grad/Reshape ^gradients/add_10_grad/Reshape_1
?
.gradients/add_10_grad/tuple/control_dependencyIdentitygradients/add_10_grad/Reshape'^gradients/add_10_grad/tuple/group_deps*#
_output_shapes
:?????????*0
_class&
$"loc:@gradients/add_10_grad/Reshape*
T0
?
0gradients/add_10_grad/tuple/control_dependency_1Identitygradients/add_10_grad/Reshape_1'^gradients/add_10_grad/tuple/group_deps*
T0*
_output_shapes
: *2
_class(
&$loc:@gradients/add_10_grad/Reshape_1
m
gradients/Neg_grad/NegNeg-gradients/mul_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
z
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
T0*
out_type0*
_output_shapes
:
y
(gradients/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
out_type0*
_output_shapes
:*
T0
?
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*'
_output_shapes
:?????????*
T0*
Tshape0
?
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
v
$gradients/logistic_loss/sub_grad/NegNeg&gradients/logistic_loss/sub_grad/Sum_1*
_output_shapes
:*
T0
?
*gradients/logistic_loss/sub_grad/Reshape_1Reshape$gradients/logistic_loss/sub_grad/Neg(gradients/logistic_loss/sub_grad/Shape_1*'
_output_shapes
:?????????*
Tshape0*
T0
?
1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp)^gradients/logistic_loss/sub_grad/Reshape+^gradients/logistic_loss/sub_grad/Reshape_1
?
9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/sub_grad/Reshape2^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*;
_class1
/-loc:@gradients/logistic_loss/sub_grad/Reshape
?
;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/sub_grad/Reshape_12^gradients/logistic_loss/sub_grad/tuple/group_deps*=
_class3
1/loc:@gradients/logistic_loss/sub_grad/Reshape_1*'
_output_shapes
:?????????*
T0
?
(gradients/logistic_loss/Log1p_grad/add/xConst8^gradients/logistic_loss_grad/tuple/control_dependency_1*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
&gradients/logistic_loss/Log1p_grad/addAdd(gradients/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0*'
_output_shapes
:?????????
?
-gradients/logistic_loss/Log1p_grad/Reciprocal
Reciprocal&gradients/logistic_loss/Log1p_grad/add*'
_output_shapes
:?????????*
T0
?
&gradients/logistic_loss/Log1p_grad/mulMul7gradients/logistic_loss_grad/tuple/control_dependency_1-gradients/logistic_loss/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:?????????
^
gradients/Sum_grad/ShapeShapeSquare*
_output_shapes
:*
out_type0*
T0
?
gradients/Sum_grad/SizeConst*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
_output_shapes
: 
?
gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
_output_shapes
: *
T0*+
_class!
loc:@gradients/Sum_grad/Shape
?
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: *
T0
?
gradients/Sum_grad/Shape_1Const*
valueB *+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
?
gradients/Sum_grad/range/startConst*
_output_shapes
: *
value	B : *
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape
?
gradients/Sum_grad/range/deltaConst*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*
_output_shapes
:*

Tidx0*+
_class!
loc:@gradients/Sum_grad/Shape
?
gradients/Sum_grad/Fill/valueConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
_output_shapes
: *
dtype0
?
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*

index_type0*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
?
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
T0*
_output_shapes
:*
N*+
_class!
loc:@gradients/Sum_grad/Shape
?
gradients/Sum_grad/Maximum/yConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:*
T0
?
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:*
T0
?
gradients/Sum_grad/ReshapeReshape.gradients/add_10_grad/tuple/control_dependency gradients/Sum_grad/DynamicStitch*
Tshape0*
T0*0
_output_shapes
:??????????????????
?
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*
T0*

Tmultiples0*(
_output_shapes
:??????????
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
?
gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
~
gradients/Mean_grad/ShapeShape%MultivariateNormalDiag_1/log_prob/add*
_output_shapes
:*
out_type0*
T0
?
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*#
_output_shapes
:?????????*
T0
?
gradients/Mean_grad/Shape_1Shape%MultivariateNormalDiag_1/log_prob/add*
out_type0*
_output_shapes
:*
T0
^
gradients/Mean_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
c
gradients/Mean_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
?
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
e
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
?
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
_
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
?
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
?
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
Truncate( *
_output_shapes
: *

SrcT0*

DstT0
?
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:?????????*
T0
?
.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikediz_1/dense_3/BiasAdd*
T0*'
_output_shapes
:?????????
?
*gradients/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual9gradients/logistic_loss/sub_grad/tuple/control_dependency.gradients/logistic_loss/Select_grad/zeros_like*'
_output_shapes
:?????????*
T0
?
,gradients/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual.gradients/logistic_loss/Select_grad/zeros_like9gradients/logistic_loss/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????
?
4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1
?
<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*
T0*'
_output_shapes
:?????????
?
>gradients/logistic_loss/Select_grad/tuple/control_dependency_1Identity,gradients/logistic_loss/Select_grad/Select_15^gradients/logistic_loss/Select_grad/tuple/group_deps*?
_class5
31loc:@gradients/logistic_loss/Select_grad/Select_1*
T0*'
_output_shapes
:?????????
{
&gradients/logistic_loss/mul_grad/ShapeShapediz_1/dense_3/BiasAdd*
T0*
_output_shapes
:*
out_type0
r
(gradients/logistic_loss/mul_grad/Shape_1Shape
zeros_like*
out_type0*
T0*
_output_shapes
:
?
6gradients/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/mul_grad/Shape(gradients/logistic_loss/mul_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
$gradients/logistic_loss/mul_grad/MulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1
zeros_like*'
_output_shapes
:?????????*
T0
?
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/Mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
?
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*
T0*'
_output_shapes
:?????????*
Tshape0
?
&gradients/logistic_loss/mul_grad/Mul_1Muldiz_1/dense_3/BiasAdd;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*'
_output_shapes
:?????????*
T0
?
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/Mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
?
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*
T0*'
_output_shapes
:?????????*
Tshape0
?
1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1
?
9gradients/logistic_loss/mul_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/mul_grad/Reshape2^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*;
_class1
/-loc:@gradients/logistic_loss/mul_grad/Reshape
?
;gradients/logistic_loss/mul_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/mul_grad/Reshape_12^gradients/logistic_loss/mul_grad/tuple/group_deps*=
_class3
1/loc:@gradients/logistic_loss/mul_grad/Reshape_1*'
_output_shapes
:?????????*
T0
?
$gradients/logistic_loss/Exp_grad/mulMul&gradients/logistic_loss/Log1p_grad/mullogistic_loss/Exp*'
_output_shapes
:?????????*
T0
z
gradients/Square_grad/ConstConst^gradients/Sum_grad/Tile*
valueB
 *   @*
dtype0*
_output_shapes
: 
w
gradients/Square_grad/MulMulsub_3gradients/Square_grad/Const*
T0*(
_output_shapes
:??????????
?
gradients/Square_grad/Mul_1Mulgradients/Sum_grad/Tilegradients/Square_grad/Mul*(
_output_shapes
:??????????*
T0
?
:gradients/MultivariateNormalDiag_1/log_prob/add_grad/ShapeShape%MultivariateNormalDiag_1/log_prob/Sum*
_output_shapes
:*
T0*
out_type0

<gradients/MultivariateNormalDiag_1/log_prob/add_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
?
Jgradients/MultivariateNormalDiag_1/log_prob/add_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/MultivariateNormalDiag_1/log_prob/add_grad/Shape<gradients/MultivariateNormalDiag_1/log_prob/add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
8gradients/MultivariateNormalDiag_1/log_prob/add_grad/SumSumgradients/Mean_grad/truedivJgradients/MultivariateNormalDiag_1/log_prob/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
<gradients/MultivariateNormalDiag_1/log_prob/add_grad/ReshapeReshape8gradients/MultivariateNormalDiag_1/log_prob/add_grad/Sum:gradients/MultivariateNormalDiag_1/log_prob/add_grad/Shape*
Tshape0*
T0*#
_output_shapes
:?????????
?
:gradients/MultivariateNormalDiag_1/log_prob/add_grad/Sum_1Sumgradients/Mean_grad/truedivLgradients/MultivariateNormalDiag_1/log_prob/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
>gradients/MultivariateNormalDiag_1/log_prob/add_grad/Reshape_1Reshape:gradients/MultivariateNormalDiag_1/log_prob/add_grad/Sum_1<gradients/MultivariateNormalDiag_1/log_prob/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
?
Egradients/MultivariateNormalDiag_1/log_prob/add_grad/tuple/group_depsNoOp=^gradients/MultivariateNormalDiag_1/log_prob/add_grad/Reshape?^gradients/MultivariateNormalDiag_1/log_prob/add_grad/Reshape_1
?
Mgradients/MultivariateNormalDiag_1/log_prob/add_grad/tuple/control_dependencyIdentity<gradients/MultivariateNormalDiag_1/log_prob/add_grad/ReshapeF^gradients/MultivariateNormalDiag_1/log_prob/add_grad/tuple/group_deps*#
_output_shapes
:?????????*
T0*O
_classE
CAloc:@gradients/MultivariateNormalDiag_1/log_prob/add_grad/Reshape
?
Ogradients/MultivariateNormalDiag_1/log_prob/add_grad/tuple/control_dependency_1Identity>gradients/MultivariateNormalDiag_1/log_prob/add_grad/Reshape_1F^gradients/MultivariateNormalDiag_1/log_prob/add_grad/tuple/group_deps*
T0*
_output_shapes
: *Q
_classG
ECloc:@gradients/MultivariateNormalDiag_1/log_prob/add_grad/Reshape_1
?
0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*'
_output_shapes
:?????????*
T0
?
,gradients/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual$gradients/logistic_loss/Exp_grad/mul0gradients/logistic_loss/Select_1_grad/zeros_like*'
_output_shapes
:?????????*
T0
?
.gradients/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients/logistic_loss/Select_1_grad/zeros_like$gradients/logistic_loss/Exp_grad/mul*
T0*'
_output_shapes
:?????????
?
6gradients/logistic_loss/Select_1_grad/tuple/group_depsNoOp-^gradients/logistic_loss/Select_1_grad/Select/^gradients/logistic_loss/Select_1_grad/Select_1
?
>gradients/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity,gradients/logistic_loss/Select_1_grad/Select7^gradients/logistic_loss/Select_1_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*?
_class5
31loc:@gradients/logistic_loss/Select_1_grad/Select
?
@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity.gradients/logistic_loss/Select_1_grad/Select_17^gradients/logistic_loss/Select_1_grad/tuple/group_deps*'
_output_shapes
:?????????*
T0*A
_class7
53loc:@gradients/logistic_loss/Select_1_grad/Select_1
i
gradients/sub_3_grad/ShapeShapeflatten/Reshape*
_output_shapes
:*
out_type0*
T0
]
gradients/sub_3_grad/Shape_1ShapeX*
_output_shapes
:*
out_type0*
T0
?
*gradients/sub_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_3_grad/Shapegradients/sub_3_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/sub_3_grad/SumSumgradients/Square_grad/Mul_1*gradients/sub_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
?
gradients/sub_3_grad/ReshapeReshapegradients/sub_3_grad/Sumgradients/sub_3_grad/Shape*(
_output_shapes
:??????????*
Tshape0*
T0
?
gradients/sub_3_grad/Sum_1Sumgradients/Square_grad/Mul_1,gradients/sub_3_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
^
gradients/sub_3_grad/NegNeggradients/sub_3_grad/Sum_1*
_output_shapes
:*
T0
?
gradients/sub_3_grad/Reshape_1Reshapegradients/sub_3_grad/Neggradients/sub_3_grad/Shape_1*
T0*(
_output_shapes
:??????????*
Tshape0
m
%gradients/sub_3_grad/tuple/group_depsNoOp^gradients/sub_3_grad/Reshape^gradients/sub_3_grad/Reshape_1
?
-gradients/sub_3_grad/tuple/control_dependencyIdentitygradients/sub_3_grad/Reshape&^gradients/sub_3_grad/tuple/group_deps*/
_class%
#!loc:@gradients/sub_3_grad/Reshape*
T0*(
_output_shapes
:??????????
?
/gradients/sub_3_grad/tuple/control_dependency_1Identitygradients/sub_3_grad/Reshape_1&^gradients/sub_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients/sub_3_grad/Reshape_1*(
_output_shapes
:??????????*
T0
?
:gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/ShapeShapeNormal_1/log_prob/sub*
_output_shapes
:*
out_type0*
T0
?
9gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/SizeConst*
value	B :*
_output_shapes
: *M
_classC
A?loc:@gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Shape*
dtype0
?
8gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/addAddMultivariateNormalDiag_1/range9gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Size*
_output_shapes
:*
T0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Shape
?
8gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/modFloorMod8gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/add9gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Size*M
_classC
A?loc:@gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Shape*
T0*
_output_shapes
:
?
<gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Shape_1Const*
valueB:*M
_classC
A?loc:@gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Shape*
_output_shapes
:*
dtype0
?
@gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/range/startConst*
value	B : *M
_classC
A?loc:@gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Shape*
dtype0*
_output_shapes
: 
?
@gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/range/deltaConst*M
_classC
A?loc:@gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Shape*
dtype0*
value	B :*
_output_shapes
: 
?
:gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/rangeRange@gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/range/start9gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Size@gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/range/delta*M
_classC
A?loc:@gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Shape*

Tidx0*
_output_shapes
:
?
?gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Fill/valueConst*
_output_shapes
: *
value	B :*M
_classC
A?loc:@gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Shape*
dtype0
?
9gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/FillFill<gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Shape_1?gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Fill/value*

index_type0*
T0*
_output_shapes
:*M
_classC
A?loc:@gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Shape
?
Bgradients/MultivariateNormalDiag_1/log_prob/Sum_grad/DynamicStitchDynamicStitch:gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/range8gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/mod:gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Shape9gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Fill*
_output_shapes
:*
T0*M
_classC
A?loc:@gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Shape*
N
?
>gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Maximum/yConst*
value	B :*
_output_shapes
: *M
_classC
A?loc:@gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Shape*
dtype0
?
<gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/MaximumMaximumBgradients/MultivariateNormalDiag_1/log_prob/Sum_grad/DynamicStitch>gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Maximum/y*
T0*
_output_shapes
:*M
_classC
A?loc:@gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Shape
?
=gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/floordivFloorDiv:gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Shape<gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Maximum*M
_classC
A?loc:@gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Shape*
T0*
_output_shapes
:
?
<gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/ReshapeReshapeMgradients/MultivariateNormalDiag_1/log_prob/add_grad/tuple/control_dependencyBgradients/MultivariateNormalDiag_1/log_prob/Sum_grad/DynamicStitch*
T0*0
_output_shapes
:??????????????????*
Tshape0
?
9gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/TileTile<gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Reshape=gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/floordiv*'
_output_shapes
:?????????d*

Tmultiples0*
T0
?
$gradients/logistic_loss/Neg_grad/NegNeg>gradients/logistic_loss/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????
|
$gradients/flatten/Reshape_grad/ShapeShapedecoder/generated_images*
T0*
out_type0*
_output_shapes
:
?
&gradients/flatten/Reshape_grad/ReshapeReshape-gradients/sub_3_grad/tuple/control_dependency$gradients/flatten/Reshape_grad/Shape*/
_output_shapes
:?????????*
Tshape0*
T0

*gradients/Normal_1/log_prob/sub_grad/ShapeShapeNormal_1/log_prob/mul*
T0*
_output_shapes
:*
out_type0
o
,gradients/Normal_1/log_prob/sub_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
?
:gradients/Normal_1/log_prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/Normal_1/log_prob/sub_grad/Shape,gradients/Normal_1/log_prob/sub_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
(gradients/Normal_1/log_prob/sub_grad/SumSum9gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Tile:gradients/Normal_1/log_prob/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
,gradients/Normal_1/log_prob/sub_grad/ReshapeReshape(gradients/Normal_1/log_prob/sub_grad/Sum*gradients/Normal_1/log_prob/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????d
?
*gradients/Normal_1/log_prob/sub_grad/Sum_1Sum9gradients/MultivariateNormalDiag_1/log_prob/Sum_grad/Tile<gradients/Normal_1/log_prob/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
~
(gradients/Normal_1/log_prob/sub_grad/NegNeg*gradients/Normal_1/log_prob/sub_grad/Sum_1*
T0*
_output_shapes
:
?
.gradients/Normal_1/log_prob/sub_grad/Reshape_1Reshape(gradients/Normal_1/log_prob/sub_grad/Neg,gradients/Normal_1/log_prob/sub_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
5gradients/Normal_1/log_prob/sub_grad/tuple/group_depsNoOp-^gradients/Normal_1/log_prob/sub_grad/Reshape/^gradients/Normal_1/log_prob/sub_grad/Reshape_1
?
=gradients/Normal_1/log_prob/sub_grad/tuple/control_dependencyIdentity,gradients/Normal_1/log_prob/sub_grad/Reshape6^gradients/Normal_1/log_prob/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Normal_1/log_prob/sub_grad/Reshape*'
_output_shapes
:?????????d
?
?gradients/Normal_1/log_prob/sub_grad/tuple/control_dependency_1Identity.gradients/Normal_1/log_prob/sub_grad/Reshape_16^gradients/Normal_1/log_prob/sub_grad/tuple/group_deps*
T0*
_output_shapes
: *A
_class7
53loc:@gradients/Normal_1/log_prob/sub_grad/Reshape_1
m
*gradients/Normal_1/log_prob/mul_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
?
,gradients/Normal_1/log_prob/mul_grad/Shape_1ShapeNormal_1/log_prob/Square*
T0*
out_type0*
_output_shapes
:
?
:gradients/Normal_1/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/Normal_1/log_prob/mul_grad/Shape,gradients/Normal_1/log_prob/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
(gradients/Normal_1/log_prob/mul_grad/MulMul=gradients/Normal_1/log_prob/sub_grad/tuple/control_dependencyNormal_1/log_prob/Square*
T0*'
_output_shapes
:?????????d
?
(gradients/Normal_1/log_prob/mul_grad/SumSum(gradients/Normal_1/log_prob/mul_grad/Mul:gradients/Normal_1/log_prob/mul_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
?
,gradients/Normal_1/log_prob/mul_grad/ReshapeReshape(gradients/Normal_1/log_prob/mul_grad/Sum*gradients/Normal_1/log_prob/mul_grad/Shape*
T0*
_output_shapes
: *
Tshape0
?
*gradients/Normal_1/log_prob/mul_grad/Mul_1MulNormal_1/log_prob/mul/x=gradients/Normal_1/log_prob/sub_grad/tuple/control_dependency*'
_output_shapes
:?????????d*
T0
?
*gradients/Normal_1/log_prob/mul_grad/Sum_1Sum*gradients/Normal_1/log_prob/mul_grad/Mul_1<gradients/Normal_1/log_prob/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
?
.gradients/Normal_1/log_prob/mul_grad/Reshape_1Reshape*gradients/Normal_1/log_prob/mul_grad/Sum_1,gradients/Normal_1/log_prob/mul_grad/Shape_1*
Tshape0*'
_output_shapes
:?????????d*
T0
?
5gradients/Normal_1/log_prob/mul_grad/tuple/group_depsNoOp-^gradients/Normal_1/log_prob/mul_grad/Reshape/^gradients/Normal_1/log_prob/mul_grad/Reshape_1
?
=gradients/Normal_1/log_prob/mul_grad/tuple/control_dependencyIdentity,gradients/Normal_1/log_prob/mul_grad/Reshape6^gradients/Normal_1/log_prob/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*?
_class5
31loc:@gradients/Normal_1/log_prob/mul_grad/Reshape
?
?gradients/Normal_1/log_prob/mul_grad/tuple/control_dependency_1Identity.gradients/Normal_1/log_prob/mul_grad/Reshape_16^gradients/Normal_1/log_prob/mul_grad/tuple/group_deps*A
_class7
53loc:@gradients/Normal_1/log_prob/mul_grad/Reshape_1*
T0*'
_output_shapes
:?????????d
?
gradients/AddNAddN<gradients/logistic_loss/Select_grad/tuple/control_dependency9gradients/logistic_loss/mul_grad/tuple/control_dependency@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1$gradients/logistic_loss/Neg_grad/Neg*
T0*'
_output_shapes
:?????????*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*
N
?
0gradients/diz_1/dense_3/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
data_formatNHWC*
_output_shapes
:*
T0
?
5gradients/diz_1/dense_3/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN1^gradients/diz_1/dense_3/BiasAdd_grad/BiasAddGrad
?
=gradients/diz_1/dense_3/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN6^gradients/diz_1/dense_3/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select
?
?gradients/diz_1/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/diz_1/dense_3/BiasAdd_grad/BiasAddGrad6^gradients/diz_1/dense_3/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*C
_class9
75loc:@gradients/diz_1/dense_3/BiasAdd_grad/BiasAddGrad
?
,gradients/decoder/layer_2/Tanh_grad/TanhGradTanhGraddecoder/layer_2/Tanh&gradients/flatten/Reshape_grad/Reshape*/
_output_shapes
:?????????*
T0
?
-gradients/Normal_1/log_prob/Square_grad/ConstConst@^gradients/Normal_1/log_prob/mul_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0*
_output_shapes
: 
?
+gradients/Normal_1/log_prob/Square_grad/MulMul%Normal_1/log_prob/standardize/truediv-gradients/Normal_1/log_prob/Square_grad/Const*
T0*'
_output_shapes
:?????????d
?
-gradients/Normal_1/log_prob/Square_grad/Mul_1Mul?gradients/Normal_1/log_prob/mul_grad/tuple/control_dependency_1+gradients/Normal_1/log_prob/Square_grad/Mul*'
_output_shapes
:?????????d*
T0
?
*gradients/diz_1/dense_3/MatMul_grad/MatMulMatMul=gradients/diz_1/dense_3/BiasAdd_grad/tuple/control_dependencydiz/dense_3/kernel/read*
transpose_a( *
transpose_b(*'
_output_shapes
:????????? *
T0
?
,gradients/diz_1/dense_3/MatMul_grad/MatMul_1MatMuldiz_1/dense_2/LeakyRelu=gradients/diz_1/dense_3/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes

: 
?
4gradients/diz_1/dense_3/MatMul_grad/tuple/group_depsNoOp+^gradients/diz_1/dense_3/MatMul_grad/MatMul-^gradients/diz_1/dense_3/MatMul_grad/MatMul_1
?
<gradients/diz_1/dense_3/MatMul_grad/tuple/control_dependencyIdentity*gradients/diz_1/dense_3/MatMul_grad/MatMul5^gradients/diz_1/dense_3/MatMul_grad/tuple/group_deps*=
_class3
1/loc:@gradients/diz_1/dense_3/MatMul_grad/MatMul*'
_output_shapes
:????????? *
T0
?
>gradients/diz_1/dense_3/MatMul_grad/tuple/control_dependency_1Identity,gradients/diz_1/dense_3/MatMul_grad/MatMul_15^gradients/diz_1/dense_3/MatMul_grad/tuple/group_deps*?
_class5
31loc:@gradients/diz_1/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

: 
?
2gradients/decoder/layer_2/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/decoder/layer_2/Tanh_grad/TanhGrad*
_output_shapes
:*
data_formatNHWC*
T0
?
7gradients/decoder/layer_2/BiasAdd_grad/tuple/group_depsNoOp3^gradients/decoder/layer_2/BiasAdd_grad/BiasAddGrad-^gradients/decoder/layer_2/Tanh_grad/TanhGrad
?
?gradients/decoder/layer_2/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/decoder/layer_2/Tanh_grad/TanhGrad8^gradients/decoder/layer_2/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/decoder/layer_2/Tanh_grad/TanhGrad*/
_output_shapes
:?????????
?
Agradients/decoder/layer_2/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/decoder/layer_2/BiasAdd_grad/BiasAddGrad8^gradients/decoder/layer_2/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*E
_class;
97loc:@gradients/decoder/layer_2/BiasAdd_grad/BiasAddGrad
?
:gradients/Normal_1/log_prob/standardize/truediv_grad/ShapeShape!Normal_1/log_prob/standardize/sub*
out_type0*
_output_shapes
:*
T0

<gradients/Normal_1/log_prob/standardize/truediv_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
?
Jgradients/Normal_1/log_prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/Normal_1/log_prob/standardize/truediv_grad/Shape<gradients/Normal_1/log_prob/standardize/truediv_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
<gradients/Normal_1/log_prob/standardize/truediv_grad/RealDivRealDiv-gradients/Normal_1/log_prob/Square_grad/Mul_1Normal_1/Identity_1*
T0*'
_output_shapes
:?????????d
?
8gradients/Normal_1/log_prob/standardize/truediv_grad/SumSum<gradients/Normal_1/log_prob/standardize/truediv_grad/RealDivJgradients/Normal_1/log_prob/standardize/truediv_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
?
<gradients/Normal_1/log_prob/standardize/truediv_grad/ReshapeReshape8gradients/Normal_1/log_prob/standardize/truediv_grad/Sum:gradients/Normal_1/log_prob/standardize/truediv_grad/Shape*
Tshape0*
T0*'
_output_shapes
:?????????d
?
8gradients/Normal_1/log_prob/standardize/truediv_grad/NegNeg!Normal_1/log_prob/standardize/sub*
T0*'
_output_shapes
:?????????d
?
>gradients/Normal_1/log_prob/standardize/truediv_grad/RealDiv_1RealDiv8gradients/Normal_1/log_prob/standardize/truediv_grad/NegNormal_1/Identity_1*
T0*'
_output_shapes
:?????????d
?
>gradients/Normal_1/log_prob/standardize/truediv_grad/RealDiv_2RealDiv>gradients/Normal_1/log_prob/standardize/truediv_grad/RealDiv_1Normal_1/Identity_1*'
_output_shapes
:?????????d*
T0
?
8gradients/Normal_1/log_prob/standardize/truediv_grad/mulMul-gradients/Normal_1/log_prob/Square_grad/Mul_1>gradients/Normal_1/log_prob/standardize/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:?????????d
?
:gradients/Normal_1/log_prob/standardize/truediv_grad/Sum_1Sum8gradients/Normal_1/log_prob/standardize/truediv_grad/mulLgradients/Normal_1/log_prob/standardize/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
>gradients/Normal_1/log_prob/standardize/truediv_grad/Reshape_1Reshape:gradients/Normal_1/log_prob/standardize/truediv_grad/Sum_1<gradients/Normal_1/log_prob/standardize/truediv_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
?
Egradients/Normal_1/log_prob/standardize/truediv_grad/tuple/group_depsNoOp=^gradients/Normal_1/log_prob/standardize/truediv_grad/Reshape?^gradients/Normal_1/log_prob/standardize/truediv_grad/Reshape_1
?
Mgradients/Normal_1/log_prob/standardize/truediv_grad/tuple/control_dependencyIdentity<gradients/Normal_1/log_prob/standardize/truediv_grad/ReshapeF^gradients/Normal_1/log_prob/standardize/truediv_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/Normal_1/log_prob/standardize/truediv_grad/Reshape*'
_output_shapes
:?????????d
?
Ogradients/Normal_1/log_prob/standardize/truediv_grad/tuple/control_dependency_1Identity>gradients/Normal_1/log_prob/standardize/truediv_grad/Reshape_1F^gradients/Normal_1/log_prob/standardize/truediv_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/Normal_1/log_prob/standardize/truediv_grad/Reshape_1*
T0*
_output_shapes
: 
?
4gradients/diz_1/dense_2/LeakyRelu_grad/LeakyReluGradLeakyReluGrad<gradients/diz_1/dense_3/MatMul_grad/tuple/control_dependencydiz_1/dense_2/BiasAdd*
alpha%??L>*'
_output_shapes
:????????? *
T0
?
5gradients/decoder/layer_2/conv2d_transpose_grad/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"         ?   
?
Dgradients/decoder/layer_2/conv2d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilter?gradients/decoder/layer_2/BiasAdd_grad/tuple/control_dependency5gradients/decoder/layer_2/conv2d_transpose_grad/Shapedecoder/LeakyRelu_1*
strides
*'
_output_shapes
:?*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*
	dilations
*
T0*
explicit_paddings
 
?
6gradients/decoder/layer_2/conv2d_transpose_grad/Conv2DConv2D?gradients/decoder/layer_2/BiasAdd_grad/tuple/control_dependencydecoder/layer_2/kernel/read*0
_output_shapes
:??????????*
paddingSAME*
data_formatNHWC*
T0*
explicit_paddings
 *
use_cudnn_on_gpu(*
strides
*
	dilations

?
@gradients/decoder/layer_2/conv2d_transpose_grad/tuple/group_depsNoOp7^gradients/decoder/layer_2/conv2d_transpose_grad/Conv2DE^gradients/decoder/layer_2/conv2d_transpose_grad/Conv2DBackpropFilter
?
Hgradients/decoder/layer_2/conv2d_transpose_grad/tuple/control_dependencyIdentityDgradients/decoder/layer_2/conv2d_transpose_grad/Conv2DBackpropFilterA^gradients/decoder/layer_2/conv2d_transpose_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/decoder/layer_2/conv2d_transpose_grad/Conv2DBackpropFilter*'
_output_shapes
:?
?
Jgradients/decoder/layer_2/conv2d_transpose_grad/tuple/control_dependency_1Identity6gradients/decoder/layer_2/conv2d_transpose_grad/Conv2DA^gradients/decoder/layer_2/conv2d_transpose_grad/tuple/group_deps*
T0*0
_output_shapes
:??????????*I
_class?
=;loc:@gradients/decoder/layer_2/conv2d_transpose_grad/Conv2D
?
6gradients/Normal_1/log_prob/standardize/sub_grad/ShapeShapeaMultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/Squeeze*
out_type0*
T0*
_output_shapes
:
{
8gradients/Normal_1/log_prob/standardize/sub_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
?
Fgradients/Normal_1/log_prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/Normal_1/log_prob/standardize/sub_grad/Shape8gradients/Normal_1/log_prob/standardize/sub_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
4gradients/Normal_1/log_prob/standardize/sub_grad/SumSumMgradients/Normal_1/log_prob/standardize/truediv_grad/tuple/control_dependencyFgradients/Normal_1/log_prob/standardize/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
?
8gradients/Normal_1/log_prob/standardize/sub_grad/ReshapeReshape4gradients/Normal_1/log_prob/standardize/sub_grad/Sum6gradients/Normal_1/log_prob/standardize/sub_grad/Shape*
T0*'
_output_shapes
:?????????d*
Tshape0
?
6gradients/Normal_1/log_prob/standardize/sub_grad/Sum_1SumMgradients/Normal_1/log_prob/standardize/truediv_grad/tuple/control_dependencyHgradients/Normal_1/log_prob/standardize/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
4gradients/Normal_1/log_prob/standardize/sub_grad/NegNeg6gradients/Normal_1/log_prob/standardize/sub_grad/Sum_1*
T0*
_output_shapes
:
?
:gradients/Normal_1/log_prob/standardize/sub_grad/Reshape_1Reshape4gradients/Normal_1/log_prob/standardize/sub_grad/Neg8gradients/Normal_1/log_prob/standardize/sub_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
?
Agradients/Normal_1/log_prob/standardize/sub_grad/tuple/group_depsNoOp9^gradients/Normal_1/log_prob/standardize/sub_grad/Reshape;^gradients/Normal_1/log_prob/standardize/sub_grad/Reshape_1
?
Igradients/Normal_1/log_prob/standardize/sub_grad/tuple/control_dependencyIdentity8gradients/Normal_1/log_prob/standardize/sub_grad/ReshapeB^gradients/Normal_1/log_prob/standardize/sub_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????d*K
_classA
?=loc:@gradients/Normal_1/log_prob/standardize/sub_grad/Reshape
?
Kgradients/Normal_1/log_prob/standardize/sub_grad/tuple/control_dependency_1Identity:gradients/Normal_1/log_prob/standardize/sub_grad/Reshape_1B^gradients/Normal_1/log_prob/standardize/sub_grad/tuple/group_deps*
_output_shapes
: *
T0*M
_classC
A?loc:@gradients/Normal_1/log_prob/standardize/sub_grad/Reshape_1
?
0gradients/diz_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients/diz_1/dense_2/LeakyRelu_grad/LeakyReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
?
5gradients/diz_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp1^gradients/diz_1/dense_2/BiasAdd_grad/BiasAddGrad5^gradients/diz_1/dense_2/LeakyRelu_grad/LeakyReluGrad
?
=gradients/diz_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity4gradients/diz_1/dense_2/LeakyRelu_grad/LeakyReluGrad6^gradients/diz_1/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:????????? *G
_class=
;9loc:@gradients/diz_1/dense_2/LeakyRelu_grad/LeakyReluGrad*
T0
?
?gradients/diz_1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/diz_1/dense_2/BiasAdd_grad/BiasAddGrad6^gradients/diz_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
: *C
_class9
75loc:@gradients/diz_1/dense_2/BiasAdd_grad/BiasAddGrad
?
0gradients/decoder/LeakyRelu_1_grad/LeakyReluGradLeakyReluGradJgradients/decoder/layer_2/conv2d_transpose_grad/tuple/control_dependency_1,decoder/batch_normalization_1/FusedBatchNorm*0
_output_shapes
:??????????*
alpha%??L>*
T0
?
vgradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/Squeeze_grad/ShapeShapevMultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul*
_output_shapes
:*
out_type0*
T0
?
xgradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/Squeeze_grad/ReshapeReshapeIgradients/Normal_1/log_prob/standardize/sub_grad/tuple/control_dependencyvgradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/Squeeze_grad/Shape*
Tshape0*
T0*+
_output_shapes
:?????????d
?
*gradients/diz_1/dense_2/MatMul_grad/MatMulMatMul=gradients/diz_1/dense_2/BiasAdd_grad/tuple/control_dependencydiz/dense_2/kernel/read*
transpose_a( *
transpose_b(*'
_output_shapes
:????????? *
T0
?
,gradients/diz_1/dense_2/MatMul_grad/MatMul_1MatMuldiz_1/dense_1/LeakyRelu=gradients/diz_1/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
_output_shapes

:  *
transpose_a(
?
4gradients/diz_1/dense_2/MatMul_grad/tuple/group_depsNoOp+^gradients/diz_1/dense_2/MatMul_grad/MatMul-^gradients/diz_1/dense_2/MatMul_grad/MatMul_1
?
<gradients/diz_1/dense_2/MatMul_grad/tuple/control_dependencyIdentity*gradients/diz_1/dense_2/MatMul_grad/MatMul5^gradients/diz_1/dense_2/MatMul_grad/tuple/group_deps*'
_output_shapes
:????????? *
T0*=
_class3
1/loc:@gradients/diz_1/dense_2/MatMul_grad/MatMul
?
>gradients/diz_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity,gradients/diz_1/dense_2/MatMul_grad/MatMul_15^gradients/diz_1/dense_2/MatMul_grad/tuple/group_deps*
T0*
_output_shapes

:  *?
_class5
31loc:@gradients/diz_1/dense_2/MatMul_grad/MatMul_1
w
gradients/zeros_like	ZerosLike.decoder/batch_normalization_1/FusedBatchNorm:1*
T0*
_output_shapes	
:?
y
gradients/zeros_like_1	ZerosLike.decoder/batch_normalization_1/FusedBatchNorm:2*
_output_shapes	
:?*
T0
y
gradients/zeros_like_2	ZerosLike.decoder/batch_normalization_1/FusedBatchNorm:3*
_output_shapes	
:?*
T0
y
gradients/zeros_like_3	ZerosLike.decoder/batch_normalization_1/FusedBatchNorm:4*
_output_shapes	
:?*
T0
?
Ngradients/decoder/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGrad0gradients/decoder/LeakyRelu_1_grad/LeakyReluGraddecoder/layer_1/BiasAdd(decoder/batch_normalization_1/gamma/read.decoder/batch_normalization_1/FusedBatchNorm:3.decoder/batch_normalization_1/FusedBatchNorm:4*
is_training(*
data_formatNHWC*
epsilon%o?:*F
_output_shapes4
2:??????????:?:?: : *
T0
?
Lgradients/decoder/batch_normalization_1/FusedBatchNorm_grad/tuple/group_depsNoOpO^gradients/decoder/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad
?
Tgradients/decoder/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependencyIdentityNgradients/decoder/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGradM^gradients/decoder/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*a
_classW
USloc:@gradients/decoder/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:??????????*
T0
?
Vgradients/decoder/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_1IdentityPgradients/decoder/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad:1M^gradients/decoder/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes	
:?*
T0*a
_classW
USloc:@gradients/decoder/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad
?
Vgradients/decoder/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_2IdentityPgradients/decoder/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad:2M^gradients/decoder/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
T0*
_output_shapes	
:?*a
_classW
USloc:@gradients/decoder/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad
?
Vgradients/decoder/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_3IdentityPgradients/decoder/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad:3M^gradients/decoder/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/decoder/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
?
Vgradients/decoder/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_4IdentityPgradients/decoder/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad:4M^gradients/decoder/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*a
_classW
USloc:@gradients/decoder/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: *
T0
?
?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/ShapeShapedMultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims*
_output_shapes
:*
out_type0*
T0
?
?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/Shape_1Const*
valueB"d      *
dtype0*
_output_shapes
:
?
?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/Shape?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/MulMulxgradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/Squeeze_grad/Reshape}MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/ExpandDims*+
_output_shapes
:?????????d*
T0
?
?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/SumSum?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/Mul?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
?
?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/ReshapeReshape?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/Sum?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/Shape*
T0*+
_output_shapes
:?????????d*
Tshape0
?
?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/Mul_1MuldMultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDimsxgradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/Squeeze_grad/Reshape*
T0*+
_output_shapes
:?????????d
?
?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/Sum_1Sum?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/Mul_1?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/Reshape_1Reshape?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/Sum_1?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:d
?
?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/tuple/group_depsNoOp?^gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/Reshape?^gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/Reshape_1
?
?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/tuple/control_dependencyIdentity?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/Reshape?^gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/tuple/group_deps*
T0*?
_class?
??loc:@gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/Reshape*+
_output_shapes
:?????????d
?
?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/tuple/control_dependency_1Identity?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/Reshape_1?^gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/tuple/group_deps*?
_class?
??loc:@gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/Reshape_1*
T0*
_output_shapes

:d
?
4gradients/diz_1/dense_1/LeakyRelu_grad/LeakyReluGradLeakyReluGrad<gradients/diz_1/dense_2/MatMul_grad/tuple/control_dependencydiz_1/dense_1/BiasAdd*'
_output_shapes
:????????? *
T0*
alpha%??L>
?
2gradients/decoder/layer_1/BiasAdd_grad/BiasAddGradBiasAddGradTgradients/decoder/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency*
_output_shapes	
:?*
T0*
data_formatNHWC
?
7gradients/decoder/layer_1/BiasAdd_grad/tuple/group_depsNoOpU^gradients/decoder/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency3^gradients/decoder/layer_1/BiasAdd_grad/BiasAddGrad
?
?gradients/decoder/layer_1/BiasAdd_grad/tuple/control_dependencyIdentityTgradients/decoder/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency8^gradients/decoder/layer_1/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/decoder/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:??????????
?
Agradients/decoder/layer_1/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/decoder/layer_1/BiasAdd_grad/BiasAddGrad8^gradients/decoder/layer_1/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@gradients/decoder/layer_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?*
T0
?
ygradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/ShapeShapeDMultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub*
out_type0*
T0*
_output_shapes
:
?
{gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/ReshapeReshape?gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/LinearOperatorDiag/solve/mul_grad/tuple/control_dependencyygradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/Shape*'
_output_shapes
:?????????d*
T0*
Tshape0
?
0gradients/diz_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients/diz_1/dense_1/LeakyRelu_grad/LeakyReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
?
5gradients/diz_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp1^gradients/diz_1/dense_1/BiasAdd_grad/BiasAddGrad5^gradients/diz_1/dense_1/LeakyRelu_grad/LeakyReluGrad
?
=gradients/diz_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity4gradients/diz_1/dense_1/LeakyRelu_grad/LeakyReluGrad6^gradients/diz_1/dense_1/BiasAdd_grad/tuple/group_deps*G
_class=
;9loc:@gradients/diz_1/dense_1/LeakyRelu_grad/LeakyReluGrad*'
_output_shapes
:????????? *
T0
?
?gradients/diz_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/diz_1/dense_1/BiasAdd_grad/BiasAddGrad6^gradients/diz_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
: *C
_class9
75loc:@gradients/diz_1/dense_1/BiasAdd_grad/BiasAddGrad
?
5gradients/decoder/layer_1/conv2d_transpose_grad/ShapeConst*%
valueB"      ?      *
dtype0*
_output_shapes
:
?
Dgradients/decoder/layer_1/conv2d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilter?gradients/decoder/layer_1/BiasAdd_grad/tuple/control_dependency5gradients/decoder/layer_1/conv2d_transpose_grad/Shapedecoder/LeakyRelu*
	dilations
*(
_output_shapes
:??*
explicit_paddings
 *
data_formatNHWC*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME
?
6gradients/decoder/layer_1/conv2d_transpose_grad/Conv2DConv2D?gradients/decoder/layer_1/BiasAdd_grad/tuple/control_dependencydecoder/layer_1/kernel/read*
T0*
	dilations
*0
_output_shapes
:??????????*
explicit_paddings
 *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides

?
@gradients/decoder/layer_1/conv2d_transpose_grad/tuple/group_depsNoOp7^gradients/decoder/layer_1/conv2d_transpose_grad/Conv2DE^gradients/decoder/layer_1/conv2d_transpose_grad/Conv2DBackpropFilter
?
Hgradients/decoder/layer_1/conv2d_transpose_grad/tuple/control_dependencyIdentityDgradients/decoder/layer_1/conv2d_transpose_grad/Conv2DBackpropFilterA^gradients/decoder/layer_1/conv2d_transpose_grad/tuple/group_deps*(
_output_shapes
:??*W
_classM
KIloc:@gradients/decoder/layer_1/conv2d_transpose_grad/Conv2DBackpropFilter*
T0
?
Jgradients/decoder/layer_1/conv2d_transpose_grad/tuple/control_dependency_1Identity6gradients/decoder/layer_1/conv2d_transpose_grad/Conv2DA^gradients/decoder/layer_1/conv2d_transpose_grad/tuple/group_deps*0
_output_shapes
:??????????*
T0*I
_class?
=;loc:@gradients/decoder/layer_1/conv2d_transpose_grad/Conv2D
?
Ygradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/ShapeShapeHencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add*
T0*
out_type0*
_output_shapes
:
?
[gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/Shape_1Const*
valueB:d*
_output_shapes
:*
dtype0
?
igradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/BroadcastGradientArgsBroadcastGradientArgsYgradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/Shape[gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
Wgradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/SumSum{gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/Reshapeigradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
?
[gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/ReshapeReshapeWgradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/SumYgradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/Shape*
Tshape0*
T0*'
_output_shapes
:?????????d
?
Ygradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/Sum_1Sum{gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/LinearOperatorDiag/solve/ExpandDims_grad/Reshapekgradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
?
Wgradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/NegNegYgradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/Sum_1*
_output_shapes
:*
T0
?
]gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1ReshapeWgradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/Neg[gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:d
?
dgradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/tuple/group_depsNoOp\^gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/Reshape^^gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1
?
lgradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependencyIdentity[gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/Reshapee^gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/tuple/group_deps*
T0*n
_classd
b`loc:@gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/Reshape*'
_output_shapes
:?????????d
?
ngradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependency_1Identity]gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1e^gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/tuple/group_deps*
T0*
_output_shapes
:d*p
_classf
dbloc:@gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/Reshape_1
?
*gradients/diz_1/dense_1/MatMul_grad/MatMulMatMul=gradients/diz_1/dense_1/BiasAdd_grad/tuple/control_dependencydiz/dense_1/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:????????? 
?
,gradients/diz_1/dense_1/MatMul_grad/MatMul_1MatMuldiz_1/dense/LeakyRelu=gradients/diz_1/dense_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:  *
transpose_a(*
transpose_b( *
T0
?
4gradients/diz_1/dense_1/MatMul_grad/tuple/group_depsNoOp+^gradients/diz_1/dense_1/MatMul_grad/MatMul-^gradients/diz_1/dense_1/MatMul_grad/MatMul_1
?
<gradients/diz_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity*gradients/diz_1/dense_1/MatMul_grad/MatMul5^gradients/diz_1/dense_1/MatMul_grad/tuple/group_deps*'
_output_shapes
:????????? *
T0*=
_class3
1/loc:@gradients/diz_1/dense_1/MatMul_grad/MatMul
?
>gradients/diz_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity,gradients/diz_1/dense_1/MatMul_grad/MatMul_15^gradients/diz_1/dense_1/MatMul_grad/tuple/group_deps*
_output_shapes

:  *?
_class5
31loc:@gradients/diz_1/dense_1/MatMul_grad/MatMul_1*
T0
?
.gradients/decoder/LeakyRelu_grad/LeakyReluGradLeakyReluGradJgradients/decoder/layer_1/conv2d_transpose_grad/tuple/control_dependency_1*decoder/batch_normalization/FusedBatchNorm*
alpha%??L>*
T0*0
_output_shapes
:??????????
?
2gradients/diz_1/dense/LeakyRelu_grad/LeakyReluGradLeakyReluGrad<gradients/diz_1/dense_1/MatMul_grad/tuple/control_dependencydiz_1/dense/BiasAdd*'
_output_shapes
:????????? *
alpha%??L>*
T0
w
gradients/zeros_like_4	ZerosLike,decoder/batch_normalization/FusedBatchNorm:1*
_output_shapes	
:?*
T0
w
gradients/zeros_like_5	ZerosLike,decoder/batch_normalization/FusedBatchNorm:2*
_output_shapes	
:?*
T0
w
gradients/zeros_like_6	ZerosLike,decoder/batch_normalization/FusedBatchNorm:3*
T0*
_output_shapes	
:?
w
gradients/zeros_like_7	ZerosLike,decoder/batch_normalization/FusedBatchNorm:4*
_output_shapes	
:?*
T0
?
Lgradients/decoder/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGrad.gradients/decoder/LeakyRelu_grad/LeakyReluGraddecoder/layer_0/BiasAdd&decoder/batch_normalization/gamma/read,decoder/batch_normalization/FusedBatchNorm:3,decoder/batch_normalization/FusedBatchNorm:4*
data_formatNHWC*F
_output_shapes4
2:??????????:?:?: : *
epsilon%o?:*
is_training(*
T0
?
Jgradients/decoder/batch_normalization/FusedBatchNorm_grad/tuple/group_depsNoOpM^gradients/decoder/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad
?
Rgradients/decoder/batch_normalization/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/decoder/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/decoder/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*0
_output_shapes
:??????????*_
_classU
SQloc:@gradients/decoder/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad
?
Tgradients/decoder/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/decoder/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/decoder/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/decoder/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:?
?
Tgradients/decoder/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/decoder/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/decoder/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes	
:?*
T0*_
_classU
SQloc:@gradients/decoder/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad
?
Tgradients/decoder/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/decoder/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/decoder/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/decoder/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
?
Tgradients/decoder/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/decoder/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/decoder/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*
_output_shapes
: *_
_classU
SQloc:@gradients/decoder/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad
?
.gradients/diz_1/dense/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients/diz_1/dense/LeakyRelu_grad/LeakyReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
?
3gradients/diz_1/dense/BiasAdd_grad/tuple/group_depsNoOp/^gradients/diz_1/dense/BiasAdd_grad/BiasAddGrad3^gradients/diz_1/dense/LeakyRelu_grad/LeakyReluGrad
?
;gradients/diz_1/dense/BiasAdd_grad/tuple/control_dependencyIdentity2gradients/diz_1/dense/LeakyRelu_grad/LeakyReluGrad4^gradients/diz_1/dense/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@gradients/diz_1/dense/LeakyRelu_grad/LeakyReluGrad*
T0*'
_output_shapes
:????????? 
?
=gradients/diz_1/dense/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/diz_1/dense/BiasAdd_grad/BiasAddGrad4^gradients/diz_1/dense/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/diz_1/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
?
2gradients/decoder/layer_0/BiasAdd_grad/BiasAddGradBiasAddGradRgradients/decoder/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency*
_output_shapes	
:?*
T0*
data_formatNHWC
?
7gradients/decoder/layer_0/BiasAdd_grad/tuple/group_depsNoOpS^gradients/decoder/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency3^gradients/decoder/layer_0/BiasAdd_grad/BiasAddGrad
?
?gradients/decoder/layer_0/BiasAdd_grad/tuple/control_dependencyIdentityRgradients/decoder/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency8^gradients/decoder/layer_0/BiasAdd_grad/tuple/group_deps*
T0*0
_output_shapes
:??????????*_
_classU
SQloc:@gradients/decoder/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad
?
Agradients/decoder/layer_0/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/decoder/layer_0/BiasAdd_grad/BiasAddGrad8^gradients/decoder/layer_0/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:?*E
_class;
97loc:@gradients/decoder/layer_0/BiasAdd_grad/BiasAddGrad*
T0
?
(gradients/diz_1/dense/MatMul_grad/MatMulMatMul;gradients/diz_1/dense/BiasAdd_grad/tuple/control_dependencydiz/dense/kernel/read*'
_output_shapes
:?????????d*
T0*
transpose_a( *
transpose_b(
?
*gradients/diz_1/dense/MatMul_grad/MatMul_1MatMul@MultivariateNormalDiag/sample/affine_linear_operator/forward/add;gradients/diz_1/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:d *
transpose_a(
?
2gradients/diz_1/dense/MatMul_grad/tuple/group_depsNoOp)^gradients/diz_1/dense/MatMul_grad/MatMul+^gradients/diz_1/dense/MatMul_grad/MatMul_1
?
:gradients/diz_1/dense/MatMul_grad/tuple/control_dependencyIdentity(gradients/diz_1/dense/MatMul_grad/MatMul3^gradients/diz_1/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:?????????d*
T0*;
_class1
/-loc:@gradients/diz_1/dense/MatMul_grad/MatMul
?
<gradients/diz_1/dense/MatMul_grad/tuple/control_dependency_1Identity*gradients/diz_1/dense/MatMul_grad/MatMul_13^gradients/diz_1/dense/MatMul_grad/tuple/group_deps*=
_class3
1/loc:@gradients/diz_1/dense/MatMul_grad/MatMul_1*
T0*
_output_shapes

:d 
?
5gradients/decoder/layer_0/conv2d_transpose_grad/ShapeConst*
_output_shapes
:*%
valueB"         d   *
dtype0
?
Dgradients/decoder/layer_0/conv2d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilter?gradients/decoder/layer_0/BiasAdd_grad/tuple/control_dependency5gradients/decoder/layer_0/conv2d_transpose_grad/Shapedecoder/Reshape*
T0*
paddingVALID*
	dilations
*
explicit_paddings
 *
strides
*
data_formatNHWC*'
_output_shapes
:?d*
use_cudnn_on_gpu(
?
6gradients/decoder/layer_0/conv2d_transpose_grad/Conv2DConv2D?gradients/decoder/layer_0/BiasAdd_grad/tuple/control_dependencydecoder/layer_0/kernel/read*
strides
*
explicit_paddings
 *
	dilations
*
paddingVALID*
use_cudnn_on_gpu(*
T0*/
_output_shapes
:?????????d*
data_formatNHWC
?
@gradients/decoder/layer_0/conv2d_transpose_grad/tuple/group_depsNoOp7^gradients/decoder/layer_0/conv2d_transpose_grad/Conv2DE^gradients/decoder/layer_0/conv2d_transpose_grad/Conv2DBackpropFilter
?
Hgradients/decoder/layer_0/conv2d_transpose_grad/tuple/control_dependencyIdentityDgradients/decoder/layer_0/conv2d_transpose_grad/Conv2DBackpropFilterA^gradients/decoder/layer_0/conv2d_transpose_grad/tuple/group_deps*W
_classM
KIloc:@gradients/decoder/layer_0/conv2d_transpose_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:?d
?
Jgradients/decoder/layer_0/conv2d_transpose_grad/tuple/control_dependency_1Identity6gradients/decoder/layer_0/conv2d_transpose_grad/Conv2DA^gradients/decoder/layer_0/conv2d_transpose_grad/tuple/group_deps*I
_class?
=;loc:@gradients/decoder/layer_0/conv2d_transpose_grad/Conv2D*
T0*/
_output_shapes
:?????????d
?
Ugradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/ShapeShapeZMultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul*
out_type0*
_output_shapes
:*
T0
?
Wgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape_1ShapeHencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add*
_output_shapes
:*
out_type0*
T0
?
egradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/ShapeWgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Sgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/SumSum:gradients/diz_1/dense/MatMul_grad/tuple/control_dependencyegradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
Wgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/ReshapeReshapeSgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/SumUgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape*'
_output_shapes
:?????????d*
Tshape0*
T0
?
Ugradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Sum_1Sum:gradients/diz_1/dense/MatMul_grad/tuple/control_dependencyggradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
?
Ygradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1ReshapeUgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Sum_1Wgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape_1*'
_output_shapes
:?????????d*
T0*
Tshape0
?
`gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/group_depsNoOpX^gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/ReshapeZ^gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1
?
hgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/control_dependencyIdentityWgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshapea^gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape*'
_output_shapes
:?????????d
?
jgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/control_dependency_1IdentityYgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1a^gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????d*l
_classb
`^loc:@gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1
?
$gradients/decoder/Reshape_grad/ShapeShapeHencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add*
out_type0*
_output_shapes
:*
T0
?
&gradients/decoder/Reshape_grad/ReshapeReshapeJgradients/decoder/layer_0/conv2d_transpose_grad/tuple/control_dependency_1$gradients/decoder/Reshape_grad/Shape*'
_output_shapes
:?????????d*
Tshape0*
T0
?
ogradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/ShapeShapeadd*
_output_shapes
:*
out_type0*
T0
?
qgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape_1Shape%MultivariateNormalDiag/sample/Reshape*
out_type0*
_output_shapes
:*
T0
?
gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/BroadcastGradientArgsBroadcastGradientArgsogradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shapeqgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
mgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/MulMulhgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/control_dependency%MultivariateNormalDiag/sample/Reshape*'
_output_shapes
:?????????d*
T0
?
mgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/SumSummgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Mulgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
?
qgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/ReshapeReshapemgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Sumogradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape*
Tshape0*'
_output_shapes
:?????????d*
T0
?
ogradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Mul_1Muladdhgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????d
?
ogradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Sum_1Sumogradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Mul_1?gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
?
sgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape_1Reshapeogradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Sum_1qgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape_1*'
_output_shapes
:?????????d*
T0*
Tshape0
?
zgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/group_depsNoOpr^gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshapet^gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape_1
?
?gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/control_dependencyIdentityqgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape{^gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/group_deps*'
_output_shapes
:?????????d*?
_classz
xvloc:@gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape*
T0
?
?gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/control_dependency_1Identitysgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape_1{^gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/group_deps*
T0*?
_class|
zxloc:@gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape_1*'
_output_shapes
:?????????d
l
gradients/add_grad/ShapeShapegen/dense_3/Softplus*
T0*
out_type0*
_output_shapes
:
]
gradients/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients/add_grad/SumSum?gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
?
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????d
?
gradients/add_grad/Sum_1Sum?gradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
?
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????d*-
_class#
!loc:@gradients/add_grad/Reshape
?
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes
: 
}
+gradients/gen/dense_3/Softplus_grad/SigmoidSigmoidgen/dense_3/BiasAdd*
T0*'
_output_shapes
:?????????d
?
'gradients/gen/dense_3/Softplus_grad/mulMul+gradients/add_grad/tuple/control_dependency+gradients/gen/dense_3/Softplus_grad/Sigmoid*'
_output_shapes
:?????????d*
T0
?
.gradients/gen/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/gen/dense_3/Softplus_grad/mul*
T0*
data_formatNHWC*
_output_shapes
:d
?
3gradients/gen/dense_3/BiasAdd_grad/tuple/group_depsNoOp/^gradients/gen/dense_3/BiasAdd_grad/BiasAddGrad(^gradients/gen/dense_3/Softplus_grad/mul
?
;gradients/gen/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/gen/dense_3/Softplus_grad/mul4^gradients/gen/dense_3/BiasAdd_grad/tuple/group_deps*:
_class0
.,loc:@gradients/gen/dense_3/Softplus_grad/mul*'
_output_shapes
:?????????d*
T0
?
=gradients/gen/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/gen/dense_3/BiasAdd_grad/BiasAddGrad4^gradients/gen/dense_3/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients/gen/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:d*
T0
?
(gradients/gen/dense_3/MatMul_grad/MatMulMatMul;gradients/gen/dense_3/BiasAdd_grad/tuple/control_dependencygen/dense_3/kernel/read*
T0*
transpose_b(*
transpose_a( *'
_output_shapes
:????????? 
?
*gradients/gen/dense_3/MatMul_grad/MatMul_1MatMulgen/dense_2/LeakyRelu;gradients/gen/dense_3/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

: d*
transpose_b( *
T0
?
2gradients/gen/dense_3/MatMul_grad/tuple/group_depsNoOp)^gradients/gen/dense_3/MatMul_grad/MatMul+^gradients/gen/dense_3/MatMul_grad/MatMul_1
?
:gradients/gen/dense_3/MatMul_grad/tuple/control_dependencyIdentity(gradients/gen/dense_3/MatMul_grad/MatMul3^gradients/gen/dense_3/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/gen/dense_3/MatMul_grad/MatMul*'
_output_shapes
:????????? 
?
<gradients/gen/dense_3/MatMul_grad/tuple/control_dependency_1Identity*gradients/gen/dense_3/MatMul_grad/MatMul_13^gradients/gen/dense_3/MatMul_grad/tuple/group_deps*
_output_shapes

: d*
T0*=
_class3
1/loc:@gradients/gen/dense_3/MatMul_grad/MatMul_1
?
2gradients/gen/dense_2/LeakyRelu_grad/LeakyReluGradLeakyReluGrad:gradients/gen/dense_3/MatMul_grad/tuple/control_dependencygen/dense_2/BiasAdd*
T0*'
_output_shapes
:????????? *
alpha%??L>
?
.gradients/gen/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients/gen/dense_2/LeakyRelu_grad/LeakyReluGrad*
_output_shapes
: *
T0*
data_formatNHWC
?
3gradients/gen/dense_2/BiasAdd_grad/tuple/group_depsNoOp/^gradients/gen/dense_2/BiasAdd_grad/BiasAddGrad3^gradients/gen/dense_2/LeakyRelu_grad/LeakyReluGrad
?
;gradients/gen/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity2gradients/gen/dense_2/LeakyRelu_grad/LeakyReluGrad4^gradients/gen/dense_2/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/gen/dense_2/LeakyRelu_grad/LeakyReluGrad*'
_output_shapes
:????????? 
?
=gradients/gen/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/gen/dense_2/BiasAdd_grad/BiasAddGrad4^gradients/gen/dense_2/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
: *A
_class7
53loc:@gradients/gen/dense_2/BiasAdd_grad/BiasAddGrad
?
(gradients/gen/dense_2/MatMul_grad/MatMulMatMul;gradients/gen/dense_2/BiasAdd_grad/tuple/control_dependencygen/dense_2/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:????????? 
?
*gradients/gen/dense_2/MatMul_grad/MatMul_1MatMulgen/dense_1/LeakyRelu;gradients/gen/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:  
?
2gradients/gen/dense_2/MatMul_grad/tuple/group_depsNoOp)^gradients/gen/dense_2/MatMul_grad/MatMul+^gradients/gen/dense_2/MatMul_grad/MatMul_1
?
:gradients/gen/dense_2/MatMul_grad/tuple/control_dependencyIdentity(gradients/gen/dense_2/MatMul_grad/MatMul3^gradients/gen/dense_2/MatMul_grad/tuple/group_deps*'
_output_shapes
:????????? *;
_class1
/-loc:@gradients/gen/dense_2/MatMul_grad/MatMul*
T0
?
<gradients/gen/dense_2/MatMul_grad/tuple/control_dependency_1Identity*gradients/gen/dense_2/MatMul_grad/MatMul_13^gradients/gen/dense_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/gen/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:  
?
2gradients/gen/dense_1/LeakyRelu_grad/LeakyReluGradLeakyReluGrad:gradients/gen/dense_2/MatMul_grad/tuple/control_dependencygen/dense_1/BiasAdd*'
_output_shapes
:????????? *
alpha%??L>*
T0
?
.gradients/gen/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients/gen/dense_1/LeakyRelu_grad/LeakyReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
?
3gradients/gen/dense_1/BiasAdd_grad/tuple/group_depsNoOp/^gradients/gen/dense_1/BiasAdd_grad/BiasAddGrad3^gradients/gen/dense_1/LeakyRelu_grad/LeakyReluGrad
?
;gradients/gen/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity2gradients/gen/dense_1/LeakyRelu_grad/LeakyReluGrad4^gradients/gen/dense_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:????????? *
T0*E
_class;
97loc:@gradients/gen/dense_1/LeakyRelu_grad/LeakyReluGrad
?
=gradients/gen/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/gen/dense_1/BiasAdd_grad/BiasAddGrad4^gradients/gen/dense_1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/gen/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
?
(gradients/gen/dense_1/MatMul_grad/MatMulMatMul;gradients/gen/dense_1/BiasAdd_grad/tuple/control_dependencygen/dense_1/kernel/read*'
_output_shapes
:????????? *
transpose_a( *
T0*
transpose_b(
?
*gradients/gen/dense_1/MatMul_grad/MatMul_1MatMulgen/dense/LeakyRelu;gradients/gen/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
_output_shapes

:  *
T0
?
2gradients/gen/dense_1/MatMul_grad/tuple/group_depsNoOp)^gradients/gen/dense_1/MatMul_grad/MatMul+^gradients/gen/dense_1/MatMul_grad/MatMul_1
?
:gradients/gen/dense_1/MatMul_grad/tuple/control_dependencyIdentity(gradients/gen/dense_1/MatMul_grad/MatMul3^gradients/gen/dense_1/MatMul_grad/tuple/group_deps*
T0*'
_output_shapes
:????????? *;
_class1
/-loc:@gradients/gen/dense_1/MatMul_grad/MatMul
?
<gradients/gen/dense_1/MatMul_grad/tuple/control_dependency_1Identity*gradients/gen/dense_1/MatMul_grad/MatMul_13^gradients/gen/dense_1/MatMul_grad/tuple/group_deps*
_output_shapes

:  *=
_class3
1/loc:@gradients/gen/dense_1/MatMul_grad/MatMul_1*
T0
?
0gradients/gen/dense/LeakyRelu_grad/LeakyReluGradLeakyReluGrad:gradients/gen/dense_1/MatMul_grad/tuple/control_dependencygen/dense/BiasAdd*
T0*'
_output_shapes
:????????? *
alpha%??L>
?
,gradients/gen/dense/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/gen/dense/LeakyRelu_grad/LeakyReluGrad*
data_formatNHWC*
T0*
_output_shapes
: 
?
1gradients/gen/dense/BiasAdd_grad/tuple/group_depsNoOp-^gradients/gen/dense/BiasAdd_grad/BiasAddGrad1^gradients/gen/dense/LeakyRelu_grad/LeakyReluGrad
?
9gradients/gen/dense/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/gen/dense/LeakyRelu_grad/LeakyReluGrad2^gradients/gen/dense/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@gradients/gen/dense/LeakyRelu_grad/LeakyReluGrad*
T0*'
_output_shapes
:????????? 
?
;gradients/gen/dense/BiasAdd_grad/tuple/control_dependency_1Identity,gradients/gen/dense/BiasAdd_grad/BiasAddGrad2^gradients/gen/dense/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/gen/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
?
&gradients/gen/dense/MatMul_grad/MatMulMatMul9gradients/gen/dense/BiasAdd_grad/tuple/control_dependencygen/dense/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:?????????f
?
(gradients/gen/dense/MatMul_grad/MatMul_1MatMul
gen/concat9gradients/gen/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
_output_shapes

:f *
transpose_a(
?
0gradients/gen/dense/MatMul_grad/tuple/group_depsNoOp'^gradients/gen/dense/MatMul_grad/MatMul)^gradients/gen/dense/MatMul_grad/MatMul_1
?
8gradients/gen/dense/MatMul_grad/tuple/control_dependencyIdentity&gradients/gen/dense/MatMul_grad/MatMul1^gradients/gen/dense/MatMul_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????f*9
_class/
-+loc:@gradients/gen/dense/MatMul_grad/MatMul
?
:gradients/gen/dense/MatMul_grad/tuple/control_dependency_1Identity(gradients/gen/dense/MatMul_grad/MatMul_11^gradients/gen/dense/MatMul_grad/tuple/group_deps*;
_class1
/-loc:@gradients/gen/dense/MatMul_grad/MatMul_1*
T0*
_output_shapes

:f 
`
gradients/gen/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
{
gradients/gen/concat_grad/modFloorModgen/concat/axisgradients/gen/concat_grad/Rank*
T0*
_output_shapes
: 
p
gradients/gen/concat_grad/ShapeShapegen/random_normal*
_output_shapes
:*
out_type0*
T0
?
 gradients/gen/concat_grad/ShapeNShapeNgen/random_normalHencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add*
T0*
out_type0*
N* 
_output_shapes
::
?
&gradients/gen/concat_grad/ConcatOffsetConcatOffsetgradients/gen/concat_grad/mod gradients/gen/concat_grad/ShapeN"gradients/gen/concat_grad/ShapeN:1* 
_output_shapes
::*
N
?
gradients/gen/concat_grad/SliceSlice8gradients/gen/dense/MatMul_grad/tuple/control_dependency&gradients/gen/concat_grad/ConcatOffset gradients/gen/concat_grad/ShapeN*
T0*'
_output_shapes
:?????????*
Index0
?
!gradients/gen/concat_grad/Slice_1Slice8gradients/gen/dense/MatMul_grad/tuple/control_dependency(gradients/gen/concat_grad/ConcatOffset:1"gradients/gen/concat_grad/ShapeN:1*'
_output_shapes
:?????????d*
Index0*
T0
x
*gradients/gen/concat_grad/tuple/group_depsNoOp ^gradients/gen/concat_grad/Slice"^gradients/gen/concat_grad/Slice_1
?
2gradients/gen/concat_grad/tuple/control_dependencyIdentitygradients/gen/concat_grad/Slice+^gradients/gen/concat_grad/tuple/group_deps*2
_class(
&$loc:@gradients/gen/concat_grad/Slice*'
_output_shapes
:?????????*
T0
?
4gradients/gen/concat_grad/tuple/control_dependency_1Identity!gradients/gen/concat_grad/Slice_1+^gradients/gen/concat_grad/tuple/group_deps*4
_class*
(&loc:@gradients/gen/concat_grad/Slice_1*'
_output_shapes
:?????????d*
T0
?
gradients/AddN_1AddNlgradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/tuple/control_dependencyjgradients/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/control_dependency_1&gradients/decoder/Reshape_grad/Reshape4gradients/gen/concat_grad/tuple/control_dependency_1*
N*
T0*'
_output_shapes
:?????????d*n
_classd
b`loc:@gradients/MultivariateNormalDiag_1/log_prob/affine_linear_operator/inverse/sub_grad/Reshape
?
]gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/ShapeShapebencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul*
T0*
out_type0*
_output_shapes
:
?
_gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape_1Shapeencoder/dense/BiasAdd*
T0*
_output_shapes
:*
out_type0
?
mgradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/BroadcastGradientArgsBroadcastGradientArgs]gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape_gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
[gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/SumSumgradients/AddN_1mgradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
_gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/ReshapeReshape[gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Sum]gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape*'
_output_shapes
:?????????d*
Tshape0*
T0
?
]gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Sum_1Sumgradients/AddN_1ogradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
agradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1Reshape]gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Sum_1_gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape_1*
Tshape0*'
_output_shapes
:?????????d*
T0
?
hgradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/group_depsNoOp`^gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshapeb^gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1
?
pgradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/control_dependencyIdentity_gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshapei^gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/group_deps*'
_output_shapes
:?????????d*r
_classh
fdloc:@gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape*
T0
?
rgradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/control_dependency_1Identityagradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1i^gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/group_deps*t
_classj
hfloc:@gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1*
T0*'
_output_shapes
:?????????d
?
wgradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/ShapeShapeencoder/dense_1/Softplus*
_output_shapes
:*
T0*
out_type0
?
ygradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape_1Shape-encoder/MultivariateNormalDiag/sample/Reshape*
out_type0*
T0*
_output_shapes
:
?
?gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/BroadcastGradientArgsBroadcastGradientArgswgradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shapeygradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
ugradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/MulMulpgradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/control_dependency-encoder/MultivariateNormalDiag/sample/Reshape*'
_output_shapes
:?????????d*
T0
?
ugradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/SumSumugradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Mul?gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:*

Tidx0
?
ygradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/ReshapeReshapeugradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Sumwgradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape*
Tshape0*'
_output_shapes
:?????????d*
T0
?
wgradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Mul_1Mulencoder/dense_1/Softpluspgradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????d
?
wgradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Sum_1Sumwgradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Mul_1?gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
{gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape_1Reshapewgradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Sum_1ygradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????d
?
?gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/group_depsNoOpz^gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape|^gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape_1
?
?gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/control_dependencyIdentityygradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape?^gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/group_deps*?
_class?
?~loc:@gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape*
T0*'
_output_shapes
:?????????d
?
?gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/control_dependency_1Identity{gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape_1?^gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/group_deps*'
_output_shapes
:?????????d*?
_class?
??loc:@gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape_1*
T0
?
0gradients/encoder/dense/BiasAdd_grad/BiasAddGradBiasAddGradrgradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/control_dependency_1*
T0*
_output_shapes
:d*
data_formatNHWC
?
5gradients/encoder/dense/BiasAdd_grad/tuple/group_depsNoOps^gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/control_dependency_11^gradients/encoder/dense/BiasAdd_grad/BiasAddGrad
?
=gradients/encoder/dense/BiasAdd_grad/tuple/control_dependencyIdentityrgradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/control_dependency_16^gradients/encoder/dense/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:?????????d*
T0*t
_classj
hfloc:@gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1
?
?gradients/encoder/dense/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/encoder/dense/BiasAdd_grad/BiasAddGrad6^gradients/encoder/dense/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:d*C
_class9
75loc:@gradients/encoder/dense/BiasAdd_grad/BiasAddGrad
?
/gradients/encoder/dense_1/Softplus_grad/SigmoidSigmoidencoder/dense_1/BiasAdd*
T0*'
_output_shapes
:?????????d
?
+gradients/encoder/dense_1/Softplus_grad/mulMul?gradients/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/control_dependency/gradients/encoder/dense_1/Softplus_grad/Sigmoid*
T0*'
_output_shapes
:?????????d
?
*gradients/encoder/dense/MatMul_grad/MatMulMatMul=gradients/encoder/dense/BiasAdd_grad/tuple/control_dependencyencoder/dense/kernel/read*
T0*
transpose_b(*(
_output_shapes
:??????????*
transpose_a( 
?
,gradients/encoder/dense/MatMul_grad/MatMul_1MatMulencoder/flatten/Reshape=gradients/encoder/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	?d*
transpose_b( *
transpose_a(*
T0
?
4gradients/encoder/dense/MatMul_grad/tuple/group_depsNoOp+^gradients/encoder/dense/MatMul_grad/MatMul-^gradients/encoder/dense/MatMul_grad/MatMul_1
?
<gradients/encoder/dense/MatMul_grad/tuple/control_dependencyIdentity*gradients/encoder/dense/MatMul_grad/MatMul5^gradients/encoder/dense/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:??????????*=
_class3
1/loc:@gradients/encoder/dense/MatMul_grad/MatMul
?
>gradients/encoder/dense/MatMul_grad/tuple/control_dependency_1Identity,gradients/encoder/dense/MatMul_grad/MatMul_15^gradients/encoder/dense/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	?d*?
_class5
31loc:@gradients/encoder/dense/MatMul_grad/MatMul_1
?
2gradients/encoder/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad+gradients/encoder/dense_1/Softplus_grad/mul*
_output_shapes
:d*
data_formatNHWC*
T0
?
7gradients/encoder/dense_1/BiasAdd_grad/tuple/group_depsNoOp3^gradients/encoder/dense_1/BiasAdd_grad/BiasAddGrad,^gradients/encoder/dense_1/Softplus_grad/mul
?
?gradients/encoder/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity+gradients/encoder/dense_1/Softplus_grad/mul8^gradients/encoder/dense_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:?????????d*>
_class4
20loc:@gradients/encoder/dense_1/Softplus_grad/mul*
T0
?
Agradients/encoder/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/encoder/dense_1/BiasAdd_grad/BiasAddGrad8^gradients/encoder/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:d*E
_class;
97loc:@gradients/encoder/dense_1/BiasAdd_grad/BiasAddGrad*
T0
?
,gradients/encoder/dense_1/MatMul_grad/MatMulMatMul?gradients/encoder/dense_1/BiasAdd_grad/tuple/control_dependencyencoder/dense_1/kernel/read*
transpose_a( *
transpose_b(*(
_output_shapes
:??????????*
T0
?
.gradients/encoder/dense_1/MatMul_grad/MatMul_1MatMulencoder/flatten/Reshape?gradients/encoder/dense_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	?d*
T0*
transpose_b( *
transpose_a(
?
6gradients/encoder/dense_1/MatMul_grad/tuple/group_depsNoOp-^gradients/encoder/dense_1/MatMul_grad/MatMul/^gradients/encoder/dense_1/MatMul_grad/MatMul_1
?
>gradients/encoder/dense_1/MatMul_grad/tuple/control_dependencyIdentity,gradients/encoder/dense_1/MatMul_grad/MatMul7^gradients/encoder/dense_1/MatMul_grad/tuple/group_deps*?
_class5
31loc:@gradients/encoder/dense_1/MatMul_grad/MatMul*(
_output_shapes
:??????????*
T0
?
@gradients/encoder/dense_1/MatMul_grad/tuple/control_dependency_1Identity.gradients/encoder/dense_1/MatMul_grad/MatMul_17^gradients/encoder/dense_1/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/encoder/dense_1/MatMul_grad/MatMul_1*
_output_shapes
:	?d
?
gradients/AddN_2AddN<gradients/encoder/dense/MatMul_grad/tuple/control_dependency>gradients/encoder/dense_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:??????????*
T0*=
_class3
1/loc:@gradients/encoder/dense/MatMul_grad/MatMul*
N
?
,gradients/encoder/flatten/Reshape_grad/ShapeShapeencoder/max_pooling2d_1/MaxPool*
T0*
out_type0*
_output_shapes
:
?
.gradients/encoder/flatten/Reshape_grad/ReshapeReshapegradients/AddN_2,gradients/encoder/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:?????????2
?
:gradients/encoder/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGradencoder/conv2d_1/Reluencoder/max_pooling2d_1/MaxPool.gradients/encoder/flatten/Reshape_grad/Reshape*
ksize
*
strides
*
data_formatNHWC*
T0*/
_output_shapes
:?????????2*
paddingVALID
?
-gradients/encoder/conv2d_1/Relu_grad/ReluGradReluGrad:gradients/encoder/max_pooling2d_1/MaxPool_grad/MaxPoolGradencoder/conv2d_1/Relu*
T0*/
_output_shapes
:?????????2
?
3gradients/encoder/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/encoder/conv2d_1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:2*
T0
?
8gradients/encoder/conv2d_1/BiasAdd_grad/tuple/group_depsNoOp4^gradients/encoder/conv2d_1/BiasAdd_grad/BiasAddGrad.^gradients/encoder/conv2d_1/Relu_grad/ReluGrad
?
@gradients/encoder/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/encoder/conv2d_1/Relu_grad/ReluGrad9^gradients/encoder/conv2d_1/BiasAdd_grad/tuple/group_deps*@
_class6
42loc:@gradients/encoder/conv2d_1/Relu_grad/ReluGrad*/
_output_shapes
:?????????2*
T0
?
Bgradients/encoder/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/encoder/conv2d_1/BiasAdd_grad/BiasAddGrad9^gradients/encoder/conv2d_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:2*
T0*F
_class<
:8loc:@gradients/encoder/conv2d_1/BiasAdd_grad/BiasAddGrad
?
-gradients/encoder/conv2d_1/Conv2D_grad/ShapeNShapeNencoder/max_pooling2d/MaxPoolencoder/conv2d_1/kernel/read*
N* 
_output_shapes
::*
T0*
out_type0
?
:gradients/encoder/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput-gradients/encoder/conv2d_1/Conv2D_grad/ShapeNencoder/conv2d_1/kernel/read@gradients/encoder/conv2d_1/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
data_formatNHWC*
T0*
use_cudnn_on_gpu(*
strides
*/
_output_shapes
:?????????*
explicit_paddings
 *
	dilations

?
;gradients/encoder/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterencoder/max_pooling2d/MaxPool/gradients/encoder/conv2d_1/Conv2D_grad/ShapeN:1@gradients/encoder/conv2d_1/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
use_cudnn_on_gpu(*
T0*
strides
*
	dilations
*
explicit_paddings
 *&
_output_shapes
:2*
data_formatNHWC
?
7gradients/encoder/conv2d_1/Conv2D_grad/tuple/group_depsNoOp<^gradients/encoder/conv2d_1/Conv2D_grad/Conv2DBackpropFilter;^gradients/encoder/conv2d_1/Conv2D_grad/Conv2DBackpropInput
?
?gradients/encoder/conv2d_1/Conv2D_grad/tuple/control_dependencyIdentity:gradients/encoder/conv2d_1/Conv2D_grad/Conv2DBackpropInput8^gradients/encoder/conv2d_1/Conv2D_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/encoder/conv2d_1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????
?
Agradients/encoder/conv2d_1/Conv2D_grad/tuple/control_dependency_1Identity;gradients/encoder/conv2d_1/Conv2D_grad/Conv2DBackpropFilter8^gradients/encoder/conv2d_1/Conv2D_grad/tuple/group_deps*&
_output_shapes
:2*
T0*N
_classD
B@loc:@gradients/encoder/conv2d_1/Conv2D_grad/Conv2DBackpropFilter
?
8gradients/encoder/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradencoder/conv2d/Reluencoder/max_pooling2d/MaxPool?gradients/encoder/conv2d_1/Conv2D_grad/tuple/control_dependency*/
_output_shapes
:?????????*
strides
*
T0*
ksize
*
data_formatNHWC*
paddingVALID
?
+gradients/encoder/conv2d/Relu_grad/ReluGradReluGrad8gradients/encoder/max_pooling2d/MaxPool_grad/MaxPoolGradencoder/conv2d/Relu*/
_output_shapes
:?????????*
T0
?
1gradients/encoder/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad+gradients/encoder/conv2d/Relu_grad/ReluGrad*
_output_shapes
:*
data_formatNHWC*
T0
?
6gradients/encoder/conv2d/BiasAdd_grad/tuple/group_depsNoOp2^gradients/encoder/conv2d/BiasAdd_grad/BiasAddGrad,^gradients/encoder/conv2d/Relu_grad/ReluGrad
?
>gradients/encoder/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity+gradients/encoder/conv2d/Relu_grad/ReluGrad7^gradients/encoder/conv2d/BiasAdd_grad/tuple/group_deps*>
_class4
20loc:@gradients/encoder/conv2d/Relu_grad/ReluGrad*/
_output_shapes
:?????????*
T0
?
@gradients/encoder/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity1gradients/encoder/conv2d/BiasAdd_grad/BiasAddGrad7^gradients/encoder/conv2d/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*D
_class:
86loc:@gradients/encoder/conv2d/BiasAdd_grad/BiasAddGrad
?
+gradients/encoder/conv2d/Conv2D_grad/ShapeNShapeNencoder/Reshapeencoder/conv2d/kernel/read*
N*
out_type0* 
_output_shapes
::*
T0
?
8gradients/encoder/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput+gradients/encoder/conv2d/Conv2D_grad/ShapeNencoder/conv2d/kernel/read>gradients/encoder/conv2d/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
paddingVALID*
	dilations
*/
_output_shapes
:?????????*
T0*
use_cudnn_on_gpu(*
explicit_paddings
 *
strides

?
9gradients/encoder/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterencoder/Reshape-gradients/encoder/conv2d/Conv2D_grad/ShapeN:1>gradients/encoder/conv2d/BiasAdd_grad/tuple/control_dependency*
T0*
explicit_paddings
 *
data_formatNHWC*
paddingVALID*&
_output_shapes
:*
use_cudnn_on_gpu(*
strides
*
	dilations

?
5gradients/encoder/conv2d/Conv2D_grad/tuple/group_depsNoOp:^gradients/encoder/conv2d/Conv2D_grad/Conv2DBackpropFilter9^gradients/encoder/conv2d/Conv2D_grad/Conv2DBackpropInput
?
=gradients/encoder/conv2d/Conv2D_grad/tuple/control_dependencyIdentity8gradients/encoder/conv2d/Conv2D_grad/Conv2DBackpropInput6^gradients/encoder/conv2d/Conv2D_grad/tuple/group_deps*
T0*/
_output_shapes
:?????????*K
_classA
?=loc:@gradients/encoder/conv2d/Conv2D_grad/Conv2DBackpropInput
?
?gradients/encoder/conv2d/Conv2D_grad/tuple/control_dependency_1Identity9gradients/encoder/conv2d/Conv2D_grad/Conv2DBackpropFilter6^gradients/encoder/conv2d/Conv2D_grad/tuple/group_deps*L
_classB
@>loc:@gradients/encoder/conv2d/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:*
T0
?
beta1_power/initial_valueConst*
_output_shapes
: *
valueB
 *fff?*3
_class)
'%loc:@decoder/batch_normalization/beta*
dtype0
?
beta1_power
VariableV2*
dtype0*
	container *
shape: *
_output_shapes
: *3
_class)
'%loc:@decoder/batch_normalization/beta*
shared_name 
?
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
_output_shapes
: *3
_class)
'%loc:@decoder/batch_normalization/beta*
validate_shape(*
T0

beta1_power/readIdentitybeta1_power*3
_class)
'%loc:@decoder/batch_normalization/beta*
T0*
_output_shapes
: 
?
beta2_power/initial_valueConst*
valueB
 *w??*
_output_shapes
: *3
_class)
'%loc:@decoder/batch_normalization/beta*
dtype0
?
beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shape: *
shared_name *3
_class)
'%loc:@decoder/batch_normalization/beta*
	container 
?
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
T0*
_output_shapes
: *3
_class)
'%loc:@decoder/batch_normalization/beta*
use_locking(

beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*3
_class)
'%loc:@decoder/batch_normalization/beta
?
,encoder/conv2d/kernel/Adam/Initializer/zerosConst*%
valueB*    *(
_class
loc:@encoder/conv2d/kernel*&
_output_shapes
:*
dtype0
?
encoder/conv2d/kernel/Adam
VariableV2*
shared_name *&
_output_shapes
:*(
_class
loc:@encoder/conv2d/kernel*
	container *
shape:*
dtype0
?
!encoder/conv2d/kernel/Adam/AssignAssignencoder/conv2d/kernel/Adam,encoder/conv2d/kernel/Adam/Initializer/zeros*(
_class
loc:@encoder/conv2d/kernel*
use_locking(*
validate_shape(*&
_output_shapes
:*
T0
?
encoder/conv2d/kernel/Adam/readIdentityencoder/conv2d/kernel/Adam*
T0*&
_output_shapes
:*(
_class
loc:@encoder/conv2d/kernel
?
.encoder/conv2d/kernel/Adam_1/Initializer/zerosConst*%
valueB*    *
dtype0*&
_output_shapes
:*(
_class
loc:@encoder/conv2d/kernel
?
encoder/conv2d/kernel/Adam_1
VariableV2*
shape:*&
_output_shapes
:*(
_class
loc:@encoder/conv2d/kernel*
	container *
shared_name *
dtype0
?
#encoder/conv2d/kernel/Adam_1/AssignAssignencoder/conv2d/kernel/Adam_1.encoder/conv2d/kernel/Adam_1/Initializer/zeros*
use_locking(*(
_class
loc:@encoder/conv2d/kernel*&
_output_shapes
:*
T0*
validate_shape(
?
!encoder/conv2d/kernel/Adam_1/readIdentityencoder/conv2d/kernel/Adam_1*&
_output_shapes
:*
T0*(
_class
loc:@encoder/conv2d/kernel
?
*encoder/conv2d/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    *&
_class
loc:@encoder/conv2d/bias*
_output_shapes
:
?
encoder/conv2d/bias/Adam
VariableV2*
	container *&
_class
loc:@encoder/conv2d/bias*
_output_shapes
:*
dtype0*
shared_name *
shape:
?
encoder/conv2d/bias/Adam/AssignAssignencoder/conv2d/bias/Adam*encoder/conv2d/bias/Adam/Initializer/zeros*
T0*
validate_shape(*&
_class
loc:@encoder/conv2d/bias*
_output_shapes
:*
use_locking(
?
encoder/conv2d/bias/Adam/readIdentityencoder/conv2d/bias/Adam*
_output_shapes
:*&
_class
loc:@encoder/conv2d/bias*
T0
?
,encoder/conv2d/bias/Adam_1/Initializer/zerosConst*&
_class
loc:@encoder/conv2d/bias*
dtype0*
valueB*    *
_output_shapes
:
?
encoder/conv2d/bias/Adam_1
VariableV2*
shared_name *
dtype0*
	container *
shape:*&
_class
loc:@encoder/conv2d/bias*
_output_shapes
:
?
!encoder/conv2d/bias/Adam_1/AssignAssignencoder/conv2d/bias/Adam_1,encoder/conv2d/bias/Adam_1/Initializer/zeros*
use_locking(*
_output_shapes
:*
T0*&
_class
loc:@encoder/conv2d/bias*
validate_shape(
?
encoder/conv2d/bias/Adam_1/readIdentityencoder/conv2d/bias/Adam_1*
T0*&
_class
loc:@encoder/conv2d/bias*
_output_shapes
:
?
>encoder/conv2d_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst**
_class 
loc:@encoder/conv2d_1/kernel*
dtype0*
_output_shapes
:*%
valueB"         2   
?
4encoder/conv2d_1/kernel/Adam/Initializer/zeros/ConstConst*
dtype0**
_class 
loc:@encoder/conv2d_1/kernel*
_output_shapes
: *
valueB
 *    
?
.encoder/conv2d_1/kernel/Adam/Initializer/zerosFill>encoder/conv2d_1/kernel/Adam/Initializer/zeros/shape_as_tensor4encoder/conv2d_1/kernel/Adam/Initializer/zeros/Const**
_class 
loc:@encoder/conv2d_1/kernel*

index_type0*
T0*&
_output_shapes
:2
?
encoder/conv2d_1/kernel/Adam
VariableV2**
_class 
loc:@encoder/conv2d_1/kernel*
	container *
shape:2*
dtype0*&
_output_shapes
:2*
shared_name 
?
#encoder/conv2d_1/kernel/Adam/AssignAssignencoder/conv2d_1/kernel/Adam.encoder/conv2d_1/kernel/Adam/Initializer/zeros*
T0**
_class 
loc:@encoder/conv2d_1/kernel*&
_output_shapes
:2*
validate_shape(*
use_locking(
?
!encoder/conv2d_1/kernel/Adam/readIdentityencoder/conv2d_1/kernel/Adam**
_class 
loc:@encoder/conv2d_1/kernel*
T0*&
_output_shapes
:2
?
@encoder/conv2d_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"         2   **
_class 
loc:@encoder/conv2d_1/kernel
?
6encoder/conv2d_1/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: **
_class 
loc:@encoder/conv2d_1/kernel*
dtype0*
valueB
 *    
?
0encoder/conv2d_1/kernel/Adam_1/Initializer/zerosFill@encoder/conv2d_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor6encoder/conv2d_1/kernel/Adam_1/Initializer/zeros/Const*
T0**
_class 
loc:@encoder/conv2d_1/kernel*

index_type0*&
_output_shapes
:2
?
encoder/conv2d_1/kernel/Adam_1
VariableV2*
shape:2*
	container *&
_output_shapes
:2**
_class 
loc:@encoder/conv2d_1/kernel*
dtype0*
shared_name 
?
%encoder/conv2d_1/kernel/Adam_1/AssignAssignencoder/conv2d_1/kernel/Adam_10encoder/conv2d_1/kernel/Adam_1/Initializer/zeros**
_class 
loc:@encoder/conv2d_1/kernel*
T0*
validate_shape(*&
_output_shapes
:2*
use_locking(
?
#encoder/conv2d_1/kernel/Adam_1/readIdentityencoder/conv2d_1/kernel/Adam_1*&
_output_shapes
:2**
_class 
loc:@encoder/conv2d_1/kernel*
T0
?
,encoder/conv2d_1/bias/Adam/Initializer/zerosConst*
valueB2*    *(
_class
loc:@encoder/conv2d_1/bias*
_output_shapes
:2*
dtype0
?
encoder/conv2d_1/bias/Adam
VariableV2*
shared_name *
	container *
dtype0*
_output_shapes
:2*(
_class
loc:@encoder/conv2d_1/bias*
shape:2
?
!encoder/conv2d_1/bias/Adam/AssignAssignencoder/conv2d_1/bias/Adam,encoder/conv2d_1/bias/Adam/Initializer/zeros*
_output_shapes
:2*(
_class
loc:@encoder/conv2d_1/bias*
use_locking(*
validate_shape(*
T0
?
encoder/conv2d_1/bias/Adam/readIdentityencoder/conv2d_1/bias/Adam*
_output_shapes
:2*
T0*(
_class
loc:@encoder/conv2d_1/bias
?
.encoder/conv2d_1/bias/Adam_1/Initializer/zerosConst*
valueB2*    *
dtype0*(
_class
loc:@encoder/conv2d_1/bias*
_output_shapes
:2
?
encoder/conv2d_1/bias/Adam_1
VariableV2*
_output_shapes
:2*
	container *
shared_name *
shape:2*(
_class
loc:@encoder/conv2d_1/bias*
dtype0
?
#encoder/conv2d_1/bias/Adam_1/AssignAssignencoder/conv2d_1/bias/Adam_1.encoder/conv2d_1/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:2*(
_class
loc:@encoder/conv2d_1/bias*
use_locking(*
T0
?
!encoder/conv2d_1/bias/Adam_1/readIdentityencoder/conv2d_1/bias/Adam_1*
T0*
_output_shapes
:2*(
_class
loc:@encoder/conv2d_1/bias
?
;encoder/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*'
_class
loc:@encoder/dense/kernel*
valueB"   d   *
dtype0
?
1encoder/dense/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    *'
_class
loc:@encoder/dense/kernel
?
+encoder/dense/kernel/Adam/Initializer/zerosFill;encoder/dense/kernel/Adam/Initializer/zeros/shape_as_tensor1encoder/dense/kernel/Adam/Initializer/zeros/Const*
T0*
_output_shapes
:	?d*'
_class
loc:@encoder/dense/kernel*

index_type0
?
encoder/dense/kernel/Adam
VariableV2*'
_class
loc:@encoder/dense/kernel*
	container *
shape:	?d*
_output_shapes
:	?d*
dtype0*
shared_name 
?
 encoder/dense/kernel/Adam/AssignAssignencoder/dense/kernel/Adam+encoder/dense/kernel/Adam/Initializer/zeros*
_output_shapes
:	?d*'
_class
loc:@encoder/dense/kernel*
validate_shape(*
T0*
use_locking(
?
encoder/dense/kernel/Adam/readIdentityencoder/dense/kernel/Adam*
T0*'
_class
loc:@encoder/dense/kernel*
_output_shapes
:	?d
?
=encoder/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"   d   *
_output_shapes
:*
dtype0*'
_class
loc:@encoder/dense/kernel
?
3encoder/dense/kernel/Adam_1/Initializer/zeros/ConstConst*'
_class
loc:@encoder/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
-encoder/dense/kernel/Adam_1/Initializer/zerosFill=encoder/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor3encoder/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_output_shapes
:	?d*'
_class
loc:@encoder/dense/kernel
?
encoder/dense/kernel/Adam_1
VariableV2*
shared_name *
_output_shapes
:	?d*'
_class
loc:@encoder/dense/kernel*
	container *
dtype0*
shape:	?d
?
"encoder/dense/kernel/Adam_1/AssignAssignencoder/dense/kernel/Adam_1-encoder/dense/kernel/Adam_1/Initializer/zeros*
T0*
validate_shape(*
_output_shapes
:	?d*'
_class
loc:@encoder/dense/kernel*
use_locking(
?
 encoder/dense/kernel/Adam_1/readIdentityencoder/dense/kernel/Adam_1*'
_class
loc:@encoder/dense/kernel*
T0*
_output_shapes
:	?d
?
)encoder/dense/bias/Adam/Initializer/zerosConst*%
_class
loc:@encoder/dense/bias*
dtype0*
_output_shapes
:d*
valueBd*    
?
encoder/dense/bias/Adam
VariableV2*
shape:d*
dtype0*
shared_name *%
_class
loc:@encoder/dense/bias*
	container *
_output_shapes
:d
?
encoder/dense/bias/Adam/AssignAssignencoder/dense/bias/Adam)encoder/dense/bias/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:d*
T0*%
_class
loc:@encoder/dense/bias
?
encoder/dense/bias/Adam/readIdentityencoder/dense/bias/Adam*
_output_shapes
:d*
T0*%
_class
loc:@encoder/dense/bias
?
+encoder/dense/bias/Adam_1/Initializer/zerosConst*
valueBd*    *%
_class
loc:@encoder/dense/bias*
_output_shapes
:d*
dtype0
?
encoder/dense/bias/Adam_1
VariableV2*
_output_shapes
:d*
shared_name *
dtype0*
shape:d*%
_class
loc:@encoder/dense/bias*
	container 
?
 encoder/dense/bias/Adam_1/AssignAssignencoder/dense/bias/Adam_1+encoder/dense/bias/Adam_1/Initializer/zeros*
_output_shapes
:d*
use_locking(*%
_class
loc:@encoder/dense/bias*
validate_shape(*
T0
?
encoder/dense/bias/Adam_1/readIdentityencoder/dense/bias/Adam_1*
T0*
_output_shapes
:d*%
_class
loc:@encoder/dense/bias
?
=encoder/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@encoder/dense_1/kernel*
_output_shapes
:*
dtype0*
valueB"   d   
?
3encoder/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *)
_class
loc:@encoder/dense_1/kernel*
dtype0
?
-encoder/dense_1/kernel/Adam/Initializer/zerosFill=encoder/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor3encoder/dense_1/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	?d*)
_class
loc:@encoder/dense_1/kernel*
T0*

index_type0
?
encoder/dense_1/kernel/Adam
VariableV2*
shape:	?d*
	container *
dtype0*
shared_name *
_output_shapes
:	?d*)
_class
loc:@encoder/dense_1/kernel
?
"encoder/dense_1/kernel/Adam/AssignAssignencoder/dense_1/kernel/Adam-encoder/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
validate_shape(*)
_class
loc:@encoder/dense_1/kernel*
_output_shapes
:	?d*
T0
?
 encoder/dense_1/kernel/Adam/readIdentityencoder/dense_1/kernel/Adam*
T0*)
_class
loc:@encoder/dense_1/kernel*
_output_shapes
:	?d
?
?encoder/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"   d   *
dtype0*
_output_shapes
:*)
_class
loc:@encoder/dense_1/kernel
?
5encoder/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *)
_class
loc:@encoder/dense_1/kernel*
dtype0*
valueB
 *    
?
/encoder/dense_1/kernel/Adam_1/Initializer/zerosFill?encoder/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor5encoder/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*
_output_shapes
:	?d*

index_type0*)
_class
loc:@encoder/dense_1/kernel
?
encoder/dense_1/kernel/Adam_1
VariableV2*)
_class
loc:@encoder/dense_1/kernel*
shared_name *
	container *
dtype0*
_output_shapes
:	?d*
shape:	?d
?
$encoder/dense_1/kernel/Adam_1/AssignAssignencoder/dense_1/kernel/Adam_1/encoder/dense_1/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	?d*
use_locking(*)
_class
loc:@encoder/dense_1/kernel*
T0
?
"encoder/dense_1/kernel/Adam_1/readIdentityencoder/dense_1/kernel/Adam_1*
T0*)
_class
loc:@encoder/dense_1/kernel*
_output_shapes
:	?d
?
+encoder/dense_1/bias/Adam/Initializer/zerosConst*'
_class
loc:@encoder/dense_1/bias*
_output_shapes
:d*
valueBd*    *
dtype0
?
encoder/dense_1/bias/Adam
VariableV2*
_output_shapes
:d*
shape:d*'
_class
loc:@encoder/dense_1/bias*
	container *
shared_name *
dtype0
?
 encoder/dense_1/bias/Adam/AssignAssignencoder/dense_1/bias/Adam+encoder/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@encoder/dense_1/bias*
use_locking(*
_output_shapes
:d*
T0
?
encoder/dense_1/bias/Adam/readIdentityencoder/dense_1/bias/Adam*
T0*'
_class
loc:@encoder/dense_1/bias*
_output_shapes
:d
?
-encoder/dense_1/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@encoder/dense_1/bias*
_output_shapes
:d*
dtype0*
valueBd*    
?
encoder/dense_1/bias/Adam_1
VariableV2*'
_class
loc:@encoder/dense_1/bias*
shared_name *
shape:d*
	container *
dtype0*
_output_shapes
:d
?
"encoder/dense_1/bias/Adam_1/AssignAssignencoder/dense_1/bias/Adam_1-encoder/dense_1/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:d*
T0*
use_locking(*'
_class
loc:@encoder/dense_1/bias
?
 encoder/dense_1/bias/Adam_1/readIdentityencoder/dense_1/bias/Adam_1*
T0*'
_class
loc:@encoder/dense_1/bias*
_output_shapes
:d
?
=decoder/layer_0/kernel/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"         d   *)
_class
loc:@decoder/layer_0/kernel*
_output_shapes
:*
dtype0
?
3decoder/layer_0/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *)
_class
loc:@decoder/layer_0/kernel*
dtype0
?
-decoder/layer_0/kernel/Adam/Initializer/zerosFill=decoder/layer_0/kernel/Adam/Initializer/zeros/shape_as_tensor3decoder/layer_0/kernel/Adam/Initializer/zeros/Const*'
_output_shapes
:?d*
T0*)
_class
loc:@decoder/layer_0/kernel*

index_type0
?
decoder/layer_0/kernel/Adam
VariableV2*
	container *
dtype0*
shared_name *'
_output_shapes
:?d*
shape:?d*)
_class
loc:@decoder/layer_0/kernel
?
"decoder/layer_0/kernel/Adam/AssignAssigndecoder/layer_0/kernel/Adam-decoder/layer_0/kernel/Adam/Initializer/zeros*)
_class
loc:@decoder/layer_0/kernel*
T0*
use_locking(*
validate_shape(*'
_output_shapes
:?d
?
 decoder/layer_0/kernel/Adam/readIdentitydecoder/layer_0/kernel/Adam*
T0*'
_output_shapes
:?d*)
_class
loc:@decoder/layer_0/kernel
?
?decoder/layer_0/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"         d   *)
_class
loc:@decoder/layer_0/kernel*
dtype0
?
5decoder/layer_0/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@decoder/layer_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
/decoder/layer_0/kernel/Adam_1/Initializer/zerosFill?decoder/layer_0/kernel/Adam_1/Initializer/zeros/shape_as_tensor5decoder/layer_0/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*'
_output_shapes
:?d*)
_class
loc:@decoder/layer_0/kernel
?
decoder/layer_0/kernel/Adam_1
VariableV2*
dtype0*
shared_name *'
_output_shapes
:?d*
shape:?d*)
_class
loc:@decoder/layer_0/kernel*
	container 
?
$decoder/layer_0/kernel/Adam_1/AssignAssigndecoder/layer_0/kernel/Adam_1/decoder/layer_0/kernel/Adam_1/Initializer/zeros*)
_class
loc:@decoder/layer_0/kernel*
use_locking(*'
_output_shapes
:?d*
T0*
validate_shape(
?
"decoder/layer_0/kernel/Adam_1/readIdentitydecoder/layer_0/kernel/Adam_1*)
_class
loc:@decoder/layer_0/kernel*
T0*'
_output_shapes
:?d
?
+decoder/layer_0/bias/Adam/Initializer/zerosConst*
_output_shapes	
:?*
dtype0*
valueB?*    *'
_class
loc:@decoder/layer_0/bias
?
decoder/layer_0/bias/Adam
VariableV2*
shared_name *
	container *
_output_shapes	
:?*
shape:?*
dtype0*'
_class
loc:@decoder/layer_0/bias
?
 decoder/layer_0/bias/Adam/AssignAssigndecoder/layer_0/bias/Adam+decoder/layer_0/bias/Adam/Initializer/zeros*
use_locking(*'
_class
loc:@decoder/layer_0/bias*
T0*
validate_shape(*
_output_shapes	
:?
?
decoder/layer_0/bias/Adam/readIdentitydecoder/layer_0/bias/Adam*
_output_shapes	
:?*
T0*'
_class
loc:@decoder/layer_0/bias
?
-decoder/layer_0/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:?*
dtype0*
valueB?*    *'
_class
loc:@decoder/layer_0/bias
?
decoder/layer_0/bias/Adam_1
VariableV2*
shared_name *
	container *
_output_shapes	
:?*
dtype0*
shape:?*'
_class
loc:@decoder/layer_0/bias
?
"decoder/layer_0/bias/Adam_1/AssignAssigndecoder/layer_0/bias/Adam_1-decoder/layer_0/bias/Adam_1/Initializer/zeros*'
_class
loc:@decoder/layer_0/bias*
T0*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
 decoder/layer_0/bias/Adam_1/readIdentitydecoder/layer_0/bias/Adam_1*'
_class
loc:@decoder/layer_0/bias*
_output_shapes	
:?*
T0
?
8decoder/batch_normalization/gamma/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*
valueB?*    *4
_class*
(&loc:@decoder/batch_normalization/gamma
?
&decoder/batch_normalization/gamma/Adam
VariableV2*
shape:?*
shared_name *4
_class*
(&loc:@decoder/batch_normalization/gamma*
dtype0*
_output_shapes	
:?*
	container 
?
-decoder/batch_normalization/gamma/Adam/AssignAssign&decoder/batch_normalization/gamma/Adam8decoder/batch_normalization/gamma/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*4
_class*
(&loc:@decoder/batch_normalization/gamma*
use_locking(*
T0
?
+decoder/batch_normalization/gamma/Adam/readIdentity&decoder/batch_normalization/gamma/Adam*
T0*4
_class*
(&loc:@decoder/batch_normalization/gamma*
_output_shapes	
:?
?
:decoder/batch_normalization/gamma/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*
valueB?*    *4
_class*
(&loc:@decoder/batch_normalization/gamma
?
(decoder/batch_normalization/gamma/Adam_1
VariableV2*
dtype0*
shared_name *
_output_shapes	
:?*4
_class*
(&loc:@decoder/batch_normalization/gamma*
shape:?*
	container 
?
/decoder/batch_normalization/gamma/Adam_1/AssignAssign(decoder/batch_normalization/gamma/Adam_1:decoder/batch_normalization/gamma/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
T0*
use_locking(*4
_class*
(&loc:@decoder/batch_normalization/gamma
?
-decoder/batch_normalization/gamma/Adam_1/readIdentity(decoder/batch_normalization/gamma/Adam_1*
_output_shapes	
:?*
T0*4
_class*
(&loc:@decoder/batch_normalization/gamma
?
7decoder/batch_normalization/beta/Adam/Initializer/zerosConst*3
_class)
'%loc:@decoder/batch_normalization/beta*
dtype0*
valueB?*    *
_output_shapes	
:?
?
%decoder/batch_normalization/beta/Adam
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*3
_class)
'%loc:@decoder/batch_normalization/beta*
	container *
shared_name 
?
,decoder/batch_normalization/beta/Adam/AssignAssign%decoder/batch_normalization/beta/Adam7decoder/batch_normalization/beta/Adam/Initializer/zeros*
T0*
use_locking(*3
_class)
'%loc:@decoder/batch_normalization/beta*
_output_shapes	
:?*
validate_shape(
?
*decoder/batch_normalization/beta/Adam/readIdentity%decoder/batch_normalization/beta/Adam*3
_class)
'%loc:@decoder/batch_normalization/beta*
T0*
_output_shapes	
:?
?
9decoder/batch_normalization/beta/Adam_1/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?*
dtype0*3
_class)
'%loc:@decoder/batch_normalization/beta
?
'decoder/batch_normalization/beta/Adam_1
VariableV2*
shape:?*
	container *
shared_name *
_output_shapes	
:?*3
_class)
'%loc:@decoder/batch_normalization/beta*
dtype0
?
.decoder/batch_normalization/beta/Adam_1/AssignAssign'decoder/batch_normalization/beta/Adam_19decoder/batch_normalization/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?*3
_class)
'%loc:@decoder/batch_normalization/beta
?
,decoder/batch_normalization/beta/Adam_1/readIdentity'decoder/batch_normalization/beta/Adam_1*
_output_shapes	
:?*3
_class)
'%loc:@decoder/batch_normalization/beta*
T0
?
=decoder/layer_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@decoder/layer_1/kernel*%
valueB"      ?      *
_output_shapes
:*
dtype0
?
3decoder/layer_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *)
_class
loc:@decoder/layer_1/kernel*
dtype0
?
-decoder/layer_1/kernel/Adam/Initializer/zerosFill=decoder/layer_1/kernel/Adam/Initializer/zeros/shape_as_tensor3decoder/layer_1/kernel/Adam/Initializer/zeros/Const*
T0*(
_output_shapes
:??*)
_class
loc:@decoder/layer_1/kernel*

index_type0
?
decoder/layer_1/kernel/Adam
VariableV2*(
_output_shapes
:??*
shape:??*)
_class
loc:@decoder/layer_1/kernel*
	container *
dtype0*
shared_name 
?
"decoder/layer_1/kernel/Adam/AssignAssigndecoder/layer_1/kernel/Adam-decoder/layer_1/kernel/Adam/Initializer/zeros*
use_locking(*)
_class
loc:@decoder/layer_1/kernel*(
_output_shapes
:??*
T0*
validate_shape(
?
 decoder/layer_1/kernel/Adam/readIdentitydecoder/layer_1/kernel/Adam*)
_class
loc:@decoder/layer_1/kernel*
T0*(
_output_shapes
:??
?
?decoder/layer_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@decoder/layer_1/kernel*
dtype0*%
valueB"      ?      *
_output_shapes
:
?
5decoder/layer_1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*)
_class
loc:@decoder/layer_1/kernel*
_output_shapes
: *
valueB
 *    
?
/decoder/layer_1/kernel/Adam_1/Initializer/zerosFill?decoder/layer_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor5decoder/layer_1/kernel/Adam_1/Initializer/zeros/Const*)
_class
loc:@decoder/layer_1/kernel*(
_output_shapes
:??*

index_type0*
T0
?
decoder/layer_1/kernel/Adam_1
VariableV2*
dtype0*
shared_name *(
_output_shapes
:??*
shape:??*)
_class
loc:@decoder/layer_1/kernel*
	container 
?
$decoder/layer_1/kernel/Adam_1/AssignAssigndecoder/layer_1/kernel/Adam_1/decoder/layer_1/kernel/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*(
_output_shapes
:??*)
_class
loc:@decoder/layer_1/kernel
?
"decoder/layer_1/kernel/Adam_1/readIdentitydecoder/layer_1/kernel/Adam_1*)
_class
loc:@decoder/layer_1/kernel*(
_output_shapes
:??*
T0
?
+decoder/layer_1/bias/Adam/Initializer/zerosConst*
_output_shapes	
:?*
dtype0*'
_class
loc:@decoder/layer_1/bias*
valueB?*    
?
decoder/layer_1/bias/Adam
VariableV2*'
_class
loc:@decoder/layer_1/bias*
shared_name *
dtype0*
_output_shapes	
:?*
	container *
shape:?
?
 decoder/layer_1/bias/Adam/AssignAssigndecoder/layer_1/bias/Adam+decoder/layer_1/bias/Adam/Initializer/zeros*
T0*'
_class
loc:@decoder/layer_1/bias*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
decoder/layer_1/bias/Adam/readIdentitydecoder/layer_1/bias/Adam*
_output_shapes	
:?*'
_class
loc:@decoder/layer_1/bias*
T0
?
-decoder/layer_1/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *'
_class
loc:@decoder/layer_1/bias*
dtype0
?
decoder/layer_1/bias/Adam_1
VariableV2*
	container *'
_class
loc:@decoder/layer_1/bias*
shared_name *
_output_shapes	
:?*
shape:?*
dtype0
?
"decoder/layer_1/bias/Adam_1/AssignAssigndecoder/layer_1/bias/Adam_1-decoder/layer_1/bias/Adam_1/Initializer/zeros*
use_locking(*
_output_shapes	
:?*
T0*
validate_shape(*'
_class
loc:@decoder/layer_1/bias
?
 decoder/layer_1/bias/Adam_1/readIdentitydecoder/layer_1/bias/Adam_1*
_output_shapes	
:?*'
_class
loc:@decoder/layer_1/bias*
T0
?
:decoder/batch_normalization_1/gamma/Adam/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *6
_class,
*(loc:@decoder/batch_normalization_1/gamma*
dtype0
?
(decoder/batch_normalization_1/gamma/Adam
VariableV2*
dtype0*6
_class,
*(loc:@decoder/batch_normalization_1/gamma*
	container *
_output_shapes	
:?*
shared_name *
shape:?
?
/decoder/batch_normalization_1/gamma/Adam/AssignAssign(decoder/batch_normalization_1/gamma/Adam:decoder/batch_normalization_1/gamma/Adam/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@decoder/batch_normalization_1/gamma*
validate_shape(*
_output_shapes	
:?
?
-decoder/batch_normalization_1/gamma/Adam/readIdentity(decoder/batch_normalization_1/gamma/Adam*
T0*6
_class,
*(loc:@decoder/batch_normalization_1/gamma*
_output_shapes	
:?
?
<decoder/batch_normalization_1/gamma/Adam_1/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?*
dtype0*6
_class,
*(loc:@decoder/batch_normalization_1/gamma
?
*decoder/batch_normalization_1/gamma/Adam_1
VariableV2*6
_class,
*(loc:@decoder/batch_normalization_1/gamma*
_output_shapes	
:?*
	container *
shape:?*
dtype0*
shared_name 
?
1decoder/batch_normalization_1/gamma/Adam_1/AssignAssign*decoder/batch_normalization_1/gamma/Adam_1<decoder/batch_normalization_1/gamma/Adam_1/Initializer/zeros*
_output_shapes	
:?*6
_class,
*(loc:@decoder/batch_normalization_1/gamma*
T0*
validate_shape(*
use_locking(
?
/decoder/batch_normalization_1/gamma/Adam_1/readIdentity*decoder/batch_normalization_1/gamma/Adam_1*
T0*6
_class,
*(loc:@decoder/batch_normalization_1/gamma*
_output_shapes	
:?
?
9decoder/batch_normalization_1/beta/Adam/Initializer/zerosConst*5
_class+
)'loc:@decoder/batch_normalization_1/beta*
dtype0*
valueB?*    *
_output_shapes	
:?
?
'decoder/batch_normalization_1/beta/Adam
VariableV2*
	container *
shared_name *
dtype0*
shape:?*5
_class+
)'loc:@decoder/batch_normalization_1/beta*
_output_shapes	
:?
?
.decoder/batch_normalization_1/beta/Adam/AssignAssign'decoder/batch_normalization_1/beta/Adam9decoder/batch_normalization_1/beta/Adam/Initializer/zeros*5
_class+
)'loc:@decoder/batch_normalization_1/beta*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:?
?
,decoder/batch_normalization_1/beta/Adam/readIdentity'decoder/batch_normalization_1/beta/Adam*5
_class+
)'loc:@decoder/batch_normalization_1/beta*
T0*
_output_shapes	
:?
?
;decoder/batch_normalization_1/beta/Adam_1/Initializer/zerosConst*
dtype0*5
_class+
)'loc:@decoder/batch_normalization_1/beta*
_output_shapes	
:?*
valueB?*    
?
)decoder/batch_normalization_1/beta/Adam_1
VariableV2*
	container *
shared_name *
shape:?*5
_class+
)'loc:@decoder/batch_normalization_1/beta*
dtype0*
_output_shapes	
:?
?
0decoder/batch_normalization_1/beta/Adam_1/AssignAssign)decoder/batch_normalization_1/beta/Adam_1;decoder/batch_normalization_1/beta/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*5
_class+
)'loc:@decoder/batch_normalization_1/beta*
T0*
_output_shapes	
:?
?
.decoder/batch_normalization_1/beta/Adam_1/readIdentity)decoder/batch_normalization_1/beta/Adam_1*
T0*5
_class+
)'loc:@decoder/batch_normalization_1/beta*
_output_shapes	
:?
?
=decoder/layer_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"         ?   *)
_class
loc:@decoder/layer_2/kernel*
_output_shapes
:*
dtype0
?
3decoder/layer_2/kernel/Adam/Initializer/zeros/ConstConst*)
_class
loc:@decoder/layer_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
-decoder/layer_2/kernel/Adam/Initializer/zerosFill=decoder/layer_2/kernel/Adam/Initializer/zeros/shape_as_tensor3decoder/layer_2/kernel/Adam/Initializer/zeros/Const*'
_output_shapes
:?*

index_type0*
T0*)
_class
loc:@decoder/layer_2/kernel
?
decoder/layer_2/kernel/Adam
VariableV2*)
_class
loc:@decoder/layer_2/kernel*
shared_name *
	container *
shape:?*'
_output_shapes
:?*
dtype0
?
"decoder/layer_2/kernel/Adam/AssignAssigndecoder/layer_2/kernel/Adam-decoder/layer_2/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*)
_class
loc:@decoder/layer_2/kernel*'
_output_shapes
:?
?
 decoder/layer_2/kernel/Adam/readIdentitydecoder/layer_2/kernel/Adam*)
_class
loc:@decoder/layer_2/kernel*
T0*'
_output_shapes
:?
?
?decoder/layer_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         ?   *
_output_shapes
:*
dtype0*)
_class
loc:@decoder/layer_2/kernel
?
5decoder/layer_2/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
dtype0*)
_class
loc:@decoder/layer_2/kernel*
valueB
 *    
?
/decoder/layer_2/kernel/Adam_1/Initializer/zerosFill?decoder/layer_2/kernel/Adam_1/Initializer/zeros/shape_as_tensor5decoder/layer_2/kernel/Adam_1/Initializer/zeros/Const*'
_output_shapes
:?*

index_type0*
T0*)
_class
loc:@decoder/layer_2/kernel
?
decoder/layer_2/kernel/Adam_1
VariableV2*
shape:?*)
_class
loc:@decoder/layer_2/kernel*
shared_name *
dtype0*
	container *'
_output_shapes
:?
?
$decoder/layer_2/kernel/Adam_1/AssignAssigndecoder/layer_2/kernel/Adam_1/decoder/layer_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*)
_class
loc:@decoder/layer_2/kernel*'
_output_shapes
:?
?
"decoder/layer_2/kernel/Adam_1/readIdentitydecoder/layer_2/kernel/Adam_1*
T0*'
_output_shapes
:?*)
_class
loc:@decoder/layer_2/kernel
?
+decoder/layer_2/bias/Adam/Initializer/zerosConst*'
_class
loc:@decoder/layer_2/bias*
valueB*    *
dtype0*
_output_shapes
:
?
decoder/layer_2/bias/Adam
VariableV2*
_output_shapes
:*
	container *
shared_name *
shape:*
dtype0*'
_class
loc:@decoder/layer_2/bias
?
 decoder/layer_2/bias/Adam/AssignAssigndecoder/layer_2/bias/Adam+decoder/layer_2/bias/Adam/Initializer/zeros*
T0*'
_class
loc:@decoder/layer_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:
?
decoder/layer_2/bias/Adam/readIdentitydecoder/layer_2/bias/Adam*
T0*
_output_shapes
:*'
_class
loc:@decoder/layer_2/bias
?
-decoder/layer_2/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*
dtype0*
valueB*    *'
_class
loc:@decoder/layer_2/bias
?
decoder/layer_2/bias/Adam_1
VariableV2*
shared_name *'
_class
loc:@decoder/layer_2/bias*
_output_shapes
:*
dtype0*
	container *
shape:
?
"decoder/layer_2/bias/Adam_1/AssignAssigndecoder/layer_2/bias/Adam_1-decoder/layer_2/bias/Adam_1/Initializer/zeros*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*'
_class
loc:@decoder/layer_2/bias
?
 decoder/layer_2/bias/Adam_1/readIdentitydecoder/layer_2/bias/Adam_1*'
_class
loc:@decoder/layer_2/bias*
_output_shapes
:*
T0
W
Adam/learning_rateConst*
valueB
 *o?:*
_output_shapes
: *
dtype0
O

Adam/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w??*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *w?+2*
dtype0*
_output_shapes
: 
?
+Adam/update_encoder/conv2d/kernel/ApplyAdam	ApplyAdamencoder/conv2d/kernelencoder/conv2d/kernel/Adamencoder/conv2d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/encoder/conv2d/Conv2D_grad/tuple/control_dependency_1*
use_locking( *(
_class
loc:@encoder/conv2d/kernel*&
_output_shapes
:*
T0*
use_nesterov( 
?
)Adam/update_encoder/conv2d/bias/ApplyAdam	ApplyAdamencoder/conv2d/biasencoder/conv2d/bias/Adamencoder/conv2d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/encoder/conv2d/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:*
use_nesterov( *
T0*
use_locking( *&
_class
loc:@encoder/conv2d/bias
?
-Adam/update_encoder/conv2d_1/kernel/ApplyAdam	ApplyAdamencoder/conv2d_1/kernelencoder/conv2d_1/kernel/Adamencoder/conv2d_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/encoder/conv2d_1/Conv2D_grad/tuple/control_dependency_1**
_class 
loc:@encoder/conv2d_1/kernel*
use_nesterov( *
use_locking( *
T0*&
_output_shapes
:2
?
+Adam/update_encoder/conv2d_1/bias/ApplyAdam	ApplyAdamencoder/conv2d_1/biasencoder/conv2d_1/bias/Adamencoder/conv2d_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonBgradients/encoder/conv2d_1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *(
_class
loc:@encoder/conv2d_1/bias*
_output_shapes
:2*
use_locking( *
T0
?
*Adam/update_encoder/dense/kernel/ApplyAdam	ApplyAdamencoder/dense/kernelencoder/dense/kernel/Adamencoder/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/encoder/dense/MatMul_grad/tuple/control_dependency_1*'
_class
loc:@encoder/dense/kernel*
T0*
use_nesterov( *
_output_shapes
:	?d*
use_locking( 
?
(Adam/update_encoder/dense/bias/ApplyAdam	ApplyAdamencoder/dense/biasencoder/dense/bias/Adamencoder/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/encoder/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes
:d*
use_nesterov( *%
_class
loc:@encoder/dense/bias
?
,Adam/update_encoder/dense_1/kernel/ApplyAdam	ApplyAdamencoder/dense_1/kernelencoder/dense_1/kernel/Adamencoder/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/encoder/dense_1/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	?d*
use_locking( *
use_nesterov( *
T0*)
_class
loc:@encoder/dense_1/kernel
?
*Adam/update_encoder/dense_1/bias/ApplyAdam	ApplyAdamencoder/dense_1/biasencoder/dense_1/bias/Adamencoder/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/encoder/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
_output_shapes
:d*
use_locking( *'
_class
loc:@encoder/dense_1/bias
?
,Adam/update_decoder/layer_0/kernel/ApplyAdam	ApplyAdamdecoder/layer_0/kerneldecoder/layer_0/kernel/Adamdecoder/layer_0/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonHgradients/decoder/layer_0/conv2d_transpose_grad/tuple/control_dependency*)
_class
loc:@decoder/layer_0/kernel*
use_nesterov( *
T0*
use_locking( *'
_output_shapes
:?d
?
*Adam/update_decoder/layer_0/bias/ApplyAdam	ApplyAdamdecoder/layer_0/biasdecoder/layer_0/bias/Adamdecoder/layer_0/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/decoder/layer_0/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:?*'
_class
loc:@decoder/layer_0/bias*
use_locking( *
T0*
use_nesterov( 
?
7Adam/update_decoder/batch_normalization/gamma/ApplyAdam	ApplyAdam!decoder/batch_normalization/gamma&decoder/batch_normalization/gamma/Adam(decoder/batch_normalization/gamma/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonTgradients/decoder/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_1*
use_locking( *
T0*
use_nesterov( *4
_class*
(&loc:@decoder/batch_normalization/gamma*
_output_shapes	
:?
?
6Adam/update_decoder/batch_normalization/beta/ApplyAdam	ApplyAdam decoder/batch_normalization/beta%decoder/batch_normalization/beta/Adam'decoder/batch_normalization/beta/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonTgradients/decoder/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_2*
_output_shapes	
:?*
use_nesterov( *3
_class)
'%loc:@decoder/batch_normalization/beta*
T0*
use_locking( 
?
,Adam/update_decoder/layer_1/kernel/ApplyAdam	ApplyAdamdecoder/layer_1/kerneldecoder/layer_1/kernel/Adamdecoder/layer_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonHgradients/decoder/layer_1/conv2d_transpose_grad/tuple/control_dependency*)
_class
loc:@decoder/layer_1/kernel*(
_output_shapes
:??*
T0*
use_nesterov( *
use_locking( 
?
*Adam/update_decoder/layer_1/bias/ApplyAdam	ApplyAdamdecoder/layer_1/biasdecoder/layer_1/bias/Adamdecoder/layer_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/decoder/layer_1/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:?*
use_locking( *
use_nesterov( *
T0*'
_class
loc:@decoder/layer_1/bias
?
9Adam/update_decoder/batch_normalization_1/gamma/ApplyAdam	ApplyAdam#decoder/batch_normalization_1/gamma(decoder/batch_normalization_1/gamma/Adam*decoder/batch_normalization_1/gamma/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonVgradients/decoder/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_1*
use_nesterov( *6
_class,
*(loc:@decoder/batch_normalization_1/gamma*
use_locking( *
T0*
_output_shapes	
:?
?
8Adam/update_decoder/batch_normalization_1/beta/ApplyAdam	ApplyAdam"decoder/batch_normalization_1/beta'decoder/batch_normalization_1/beta/Adam)decoder/batch_normalization_1/beta/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonVgradients/decoder/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_2*
_output_shapes	
:?*
use_nesterov( *
use_locking( *5
_class+
)'loc:@decoder/batch_normalization_1/beta*
T0
?
,Adam/update_decoder/layer_2/kernel/ApplyAdam	ApplyAdamdecoder/layer_2/kerneldecoder/layer_2/kernel/Adamdecoder/layer_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonHgradients/decoder/layer_2/conv2d_transpose_grad/tuple/control_dependency*
use_locking( *'
_output_shapes
:?*
use_nesterov( *
T0*)
_class
loc:@decoder/layer_2/kernel
?
*Adam/update_decoder/layer_2/bias/ApplyAdam	ApplyAdamdecoder/layer_2/biasdecoder/layer_2/bias/Adamdecoder/layer_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/decoder/layer_2/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *'
_class
loc:@decoder/layer_2/bias*
_output_shapes
:*
use_locking( *
T0
?
Adam/mulMulbeta1_power/read
Adam/beta17^Adam/update_decoder/batch_normalization/beta/ApplyAdam8^Adam/update_decoder/batch_normalization/gamma/ApplyAdam9^Adam/update_decoder/batch_normalization_1/beta/ApplyAdam:^Adam/update_decoder/batch_normalization_1/gamma/ApplyAdam+^Adam/update_decoder/layer_0/bias/ApplyAdam-^Adam/update_decoder/layer_0/kernel/ApplyAdam+^Adam/update_decoder/layer_1/bias/ApplyAdam-^Adam/update_decoder/layer_1/kernel/ApplyAdam+^Adam/update_decoder/layer_2/bias/ApplyAdam-^Adam/update_decoder/layer_2/kernel/ApplyAdam*^Adam/update_encoder/conv2d/bias/ApplyAdam,^Adam/update_encoder/conv2d/kernel/ApplyAdam,^Adam/update_encoder/conv2d_1/bias/ApplyAdam.^Adam/update_encoder/conv2d_1/kernel/ApplyAdam)^Adam/update_encoder/dense/bias/ApplyAdam+^Adam/update_encoder/dense/kernel/ApplyAdam+^Adam/update_encoder/dense_1/bias/ApplyAdam-^Adam/update_encoder/dense_1/kernel/ApplyAdam*
T0*
_output_shapes
: *3
_class)
'%loc:@decoder/batch_normalization/beta
?
Adam/AssignAssignbeta1_powerAdam/mul*
_output_shapes
: *
validate_shape(*
T0*
use_locking( *3
_class)
'%loc:@decoder/batch_normalization/beta
?

Adam/mul_1Mulbeta2_power/read
Adam/beta27^Adam/update_decoder/batch_normalization/beta/ApplyAdam8^Adam/update_decoder/batch_normalization/gamma/ApplyAdam9^Adam/update_decoder/batch_normalization_1/beta/ApplyAdam:^Adam/update_decoder/batch_normalization_1/gamma/ApplyAdam+^Adam/update_decoder/layer_0/bias/ApplyAdam-^Adam/update_decoder/layer_0/kernel/ApplyAdam+^Adam/update_decoder/layer_1/bias/ApplyAdam-^Adam/update_decoder/layer_1/kernel/ApplyAdam+^Adam/update_decoder/layer_2/bias/ApplyAdam-^Adam/update_decoder/layer_2/kernel/ApplyAdam*^Adam/update_encoder/conv2d/bias/ApplyAdam,^Adam/update_encoder/conv2d/kernel/ApplyAdam,^Adam/update_encoder/conv2d_1/bias/ApplyAdam.^Adam/update_encoder/conv2d_1/kernel/ApplyAdam)^Adam/update_encoder/dense/bias/ApplyAdam+^Adam/update_encoder/dense/kernel/ApplyAdam+^Adam/update_encoder/dense_1/bias/ApplyAdam-^Adam/update_encoder/dense_1/kernel/ApplyAdam*
T0*3
_class)
'%loc:@decoder/batch_normalization/beta*
_output_shapes
: 
?
Adam/Assign_1Assignbeta2_power
Adam/mul_1*3
_class)
'%loc:@decoder/batch_normalization/beta*
use_locking( *
T0*
validate_shape(*
_output_shapes
: 
?
AdamNoOp^Adam/Assign^Adam/Assign_17^Adam/update_decoder/batch_normalization/beta/ApplyAdam8^Adam/update_decoder/batch_normalization/gamma/ApplyAdam9^Adam/update_decoder/batch_normalization_1/beta/ApplyAdam:^Adam/update_decoder/batch_normalization_1/gamma/ApplyAdam+^Adam/update_decoder/layer_0/bias/ApplyAdam-^Adam/update_decoder/layer_0/kernel/ApplyAdam+^Adam/update_decoder/layer_1/bias/ApplyAdam-^Adam/update_decoder/layer_1/kernel/ApplyAdam+^Adam/update_decoder/layer_2/bias/ApplyAdam-^Adam/update_decoder/layer_2/kernel/ApplyAdam*^Adam/update_encoder/conv2d/bias/ApplyAdam,^Adam/update_encoder/conv2d/kernel/ApplyAdam,^Adam/update_encoder/conv2d_1/bias/ApplyAdam.^Adam/update_encoder/conv2d_1/kernel/ApplyAdam)^Adam/update_encoder/dense/bias/ApplyAdam+^Adam/update_encoder/dense/kernel/ApplyAdam+^Adam/update_encoder/dense_1/bias/ApplyAdam-^Adam/update_encoder/dense_1/kernel/ApplyAdam
T
gradients_1/ShapeConst*
_output_shapes
: *
valueB *
dtype0
Z
gradients_1/grad_ys_0Const*
dtype0*
valueB
 *  ??*
_output_shapes
: 
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
_output_shapes
: *

index_type0*
T0
B
'gradients_1/add_3_grad/tuple/group_depsNoOp^gradients_1/Fill
?
/gradients_1/add_3_grad/tuple/control_dependencyIdentitygradients_1/Fill(^gradients_1/add_3_grad/tuple/group_deps*
T0*
_output_shapes
: *#
_class
loc:@gradients_1/Fill
?
1gradients_1/add_3_grad/tuple/control_dependency_1Identitygradients_1/Fill(^gradients_1/add_3_grad/tuple/group_deps*
T0*
_output_shapes
: *#
_class
loc:@gradients_1/Fill
v
%gradients_1/Mean_2_grad/Reshape/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
?
gradients_1/Mean_2_grad/ReshapeReshape/gradients_1/add_3_grad/tuple/control_dependency%gradients_1/Mean_2_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
b
gradients_1/Mean_2_grad/ShapeShapeNeg_2*
T0*
_output_shapes
:*
out_type0
?
gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:?????????
d
gradients_1/Mean_2_grad/Shape_1ShapeNeg_2*
T0*
_output_shapes
:*
out_type0
b
gradients_1/Mean_2_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_1/Mean_2_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
?
gradients_1/Mean_2_grad/ProdProdgradients_1/Mean_2_grad/Shape_1gradients_1/Mean_2_grad/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
i
gradients_1/Mean_2_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
?
gradients_1/Mean_2_grad/Prod_1Prodgradients_1/Mean_2_grad/Shape_2gradients_1/Mean_2_grad/Const_1*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
c
!gradients_1/Mean_2_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
?
gradients_1/Mean_2_grad/MaximumMaximumgradients_1/Mean_2_grad/Prod_1!gradients_1/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 
?
 gradients_1/Mean_2_grad/floordivFloorDivgradients_1/Mean_2_grad/Prodgradients_1/Mean_2_grad/Maximum*
T0*
_output_shapes
: 
?
gradients_1/Mean_2_grad/CastCast gradients_1/Mean_2_grad/floordiv*

SrcT0*

DstT0*
Truncate( *
_output_shapes
: 
?
gradients_1/Mean_2_grad/truedivRealDivgradients_1/Mean_2_grad/Tilegradients_1/Mean_2_grad/Cast*
T0*'
_output_shapes
:?????????
v
%gradients_1/Mean_3_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
?
gradients_1/Mean_3_grad/ReshapeReshape1gradients_1/add_3_grad/tuple/control_dependency_1%gradients_1/Mean_3_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
b
gradients_1/Mean_3_grad/ShapeShapeNeg_3*
T0*
out_type0*
_output_shapes
:
?
gradients_1/Mean_3_grad/TileTilegradients_1/Mean_3_grad/Reshapegradients_1/Mean_3_grad/Shape*
T0*

Tmultiples0*'
_output_shapes
:?????????
d
gradients_1/Mean_3_grad/Shape_1ShapeNeg_3*
out_type0*
T0*
_output_shapes
:
b
gradients_1/Mean_3_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
g
gradients_1/Mean_3_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
gradients_1/Mean_3_grad/ProdProdgradients_1/Mean_3_grad/Shape_1gradients_1/Mean_3_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
i
gradients_1/Mean_3_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
?
gradients_1/Mean_3_grad/Prod_1Prodgradients_1/Mean_3_grad/Shape_2gradients_1/Mean_3_grad/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
c
!gradients_1/Mean_3_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
?
gradients_1/Mean_3_grad/MaximumMaximumgradients_1/Mean_3_grad/Prod_1!gradients_1/Mean_3_grad/Maximum/y*
_output_shapes
: *
T0
?
 gradients_1/Mean_3_grad/floordivFloorDivgradients_1/Mean_3_grad/Prodgradients_1/Mean_3_grad/Maximum*
_output_shapes
: *
T0
?
gradients_1/Mean_3_grad/CastCast gradients_1/Mean_3_grad/floordiv*
_output_shapes
: *
Truncate( *

SrcT0*

DstT0
?
gradients_1/Mean_3_grad/truedivRealDivgradients_1/Mean_3_grad/Tilegradients_1/Mean_3_grad/Cast*'
_output_shapes
:?????????*
T0
t
gradients_1/Neg_2_grad/NegNeggradients_1/Mean_2_grad/truediv*
T0*'
_output_shapes
:?????????
t
gradients_1/Neg_3_grad/NegNeggradients_1/Mean_3_grad/truediv*'
_output_shapes
:?????????*
T0
?
gradients_1/Log_grad/Reciprocal
Reciprocaladd_1^gradients_1/Neg_2_grad/Neg*'
_output_shapes
:?????????*
T0
?
gradients_1/Log_grad/mulMulgradients_1/Neg_2_grad/Neggradients_1/Log_grad/Reciprocal*'
_output_shapes
:?????????*
T0
?
!gradients_1/Log_1_grad/Reciprocal
Reciprocaladd_2^gradients_1/Neg_3_grad/Neg*'
_output_shapes
:?????????*
T0
?
gradients_1/Log_1_grad/mulMulgradients_1/Neg_3_grad/Neg!gradients_1/Log_1_grad/Reciprocal*'
_output_shapes
:?????????*
T0
c
gradients_1/add_1_grad/ShapeShapeSigmoid*
T0*
_output_shapes
:*
out_type0
a
gradients_1/add_1_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
?
,gradients_1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_1_grad/Shapegradients_1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_1/add_1_grad/SumSumgradients_1/Log_grad/mul,gradients_1/add_1_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
?
gradients_1/add_1_grad/ReshapeReshapegradients_1/add_1_grad/Sumgradients_1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients_1/add_1_grad/Sum_1Sumgradients_1/Log_grad/mul.gradients_1/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
 gradients_1/add_1_grad/Reshape_1Reshapegradients_1/add_1_grad/Sum_1gradients_1/add_1_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
s
'gradients_1/add_1_grad/tuple/group_depsNoOp^gradients_1/add_1_grad/Reshape!^gradients_1/add_1_grad/Reshape_1
?
/gradients_1/add_1_grad/tuple/control_dependencyIdentitygradients_1/add_1_grad/Reshape(^gradients_1/add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_1_grad/Reshape*
T0*'
_output_shapes
:?????????
?
1gradients_1/add_1_grad/tuple/control_dependency_1Identity gradients_1/add_1_grad/Reshape_1(^gradients_1/add_1_grad/tuple/group_deps*
_output_shapes
: *3
_class)
'%loc:@gradients_1/add_1_grad/Reshape_1*
T0
_
gradients_1/add_2_grad/ShapeShapesub*
out_type0*
_output_shapes
:*
T0
a
gradients_1/add_2_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
?
,gradients_1/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_2_grad/Shapegradients_1/add_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_1/add_2_grad/SumSumgradients_1/Log_1_grad/mul,gradients_1/add_2_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
?
gradients_1/add_2_grad/ReshapeReshapegradients_1/add_2_grad/Sumgradients_1/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients_1/add_2_grad/Sum_1Sumgradients_1/Log_1_grad/mul.gradients_1/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
?
 gradients_1/add_2_grad/Reshape_1Reshapegradients_1/add_2_grad/Sum_1gradients_1/add_2_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
s
'gradients_1/add_2_grad/tuple/group_depsNoOp^gradients_1/add_2_grad/Reshape!^gradients_1/add_2_grad/Reshape_1
?
/gradients_1/add_2_grad/tuple/control_dependencyIdentitygradients_1/add_2_grad/Reshape(^gradients_1/add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_2_grad/Reshape*'
_output_shapes
:?????????*
T0
?
1gradients_1/add_2_grad/tuple/control_dependency_1Identity gradients_1/add_2_grad/Reshape_1(^gradients_1/add_2_grad/tuple/group_deps*
_output_shapes
: *3
_class)
'%loc:@gradients_1/add_2_grad/Reshape_1*
T0
?
$gradients_1/Sigmoid_grad/SigmoidGradSigmoidGradSigmoid/gradients_1/add_1_grad/tuple/control_dependency*'
_output_shapes
:?????????*
T0
]
gradients_1/sub_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
e
gradients_1/sub_grad/Shape_1Shape	Sigmoid_1*
out_type0*
T0*
_output_shapes
:
?
*gradients_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_grad/Shapegradients_1/sub_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_1/sub_grad/SumSum/gradients_1/add_2_grad/tuple/control_dependency*gradients_1/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
gradients_1/sub_grad/ReshapeReshapegradients_1/sub_grad/Sumgradients_1/sub_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
?
gradients_1/sub_grad/Sum_1Sum/gradients_1/add_2_grad/tuple/control_dependency,gradients_1/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
^
gradients_1/sub_grad/NegNeggradients_1/sub_grad/Sum_1*
T0*
_output_shapes
:
?
gradients_1/sub_grad/Reshape_1Reshapegradients_1/sub_grad/Neggradients_1/sub_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:?????????
m
%gradients_1/sub_grad/tuple/group_depsNoOp^gradients_1/sub_grad/Reshape^gradients_1/sub_grad/Reshape_1
?
-gradients_1/sub_grad/tuple/control_dependencyIdentitygradients_1/sub_grad/Reshape&^gradients_1/sub_grad/tuple/group_deps*/
_class%
#!loc:@gradients_1/sub_grad/Reshape*
T0*
_output_shapes
: 
?
/gradients_1/sub_grad/tuple/control_dependency_1Identitygradients_1/sub_grad/Reshape_1&^gradients_1/sub_grad/tuple/group_deps*'
_output_shapes
:?????????*
T0*1
_class'
%#loc:@gradients_1/sub_grad/Reshape_1
?
0gradients_1/diz/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients_1/Sigmoid_grad/SigmoidGrad*
_output_shapes
:*
data_formatNHWC*
T0
?
5gradients_1/diz/dense_3/BiasAdd_grad/tuple/group_depsNoOp%^gradients_1/Sigmoid_grad/SigmoidGrad1^gradients_1/diz/dense_3/BiasAdd_grad/BiasAddGrad
?
=gradients_1/diz/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity$gradients_1/Sigmoid_grad/SigmoidGrad6^gradients_1/diz/dense_3/BiasAdd_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients_1/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:?????????
?
?gradients_1/diz/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients_1/diz/dense_3/BiasAdd_grad/BiasAddGrad6^gradients_1/diz/dense_3/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*C
_class9
75loc:@gradients_1/diz/dense_3/BiasAdd_grad/BiasAddGrad*
T0
?
&gradients_1/Sigmoid_1_grad/SigmoidGradSigmoidGrad	Sigmoid_1/gradients_1/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
*gradients_1/diz/dense_3/MatMul_grad/MatMulMatMul=gradients_1/diz/dense_3/BiasAdd_grad/tuple/control_dependencydiz/dense_3/kernel/read*
T0*
transpose_b(*'
_output_shapes
:????????? *
transpose_a( 
?
,gradients_1/diz/dense_3/MatMul_grad/MatMul_1MatMuldiz/dense_2/LeakyRelu=gradients_1/diz/dense_3/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

: *
transpose_b( *
transpose_a(
?
4gradients_1/diz/dense_3/MatMul_grad/tuple/group_depsNoOp+^gradients_1/diz/dense_3/MatMul_grad/MatMul-^gradients_1/diz/dense_3/MatMul_grad/MatMul_1
?
<gradients_1/diz/dense_3/MatMul_grad/tuple/control_dependencyIdentity*gradients_1/diz/dense_3/MatMul_grad/MatMul5^gradients_1/diz/dense_3/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/diz/dense_3/MatMul_grad/MatMul*'
_output_shapes
:????????? 
?
>gradients_1/diz/dense_3/MatMul_grad/tuple/control_dependency_1Identity,gradients_1/diz/dense_3/MatMul_grad/MatMul_15^gradients_1/diz/dense_3/MatMul_grad/tuple/group_deps*?
_class5
31loc:@gradients_1/diz/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

: 
?
2gradients_1/diz_1/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad&gradients_1/Sigmoid_1_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
?
7gradients_1/diz_1/dense_3/BiasAdd_grad/tuple/group_depsNoOp'^gradients_1/Sigmoid_1_grad/SigmoidGrad3^gradients_1/diz_1/dense_3/BiasAdd_grad/BiasAddGrad
?
?gradients_1/diz_1/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity&gradients_1/Sigmoid_1_grad/SigmoidGrad8^gradients_1/diz_1/dense_3/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:?????????*9
_class/
-+loc:@gradients_1/Sigmoid_1_grad/SigmoidGrad*
T0
?
Agradients_1/diz_1/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_1/diz_1/dense_3/BiasAdd_grad/BiasAddGrad8^gradients_1/diz_1/dense_3/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*E
_class;
97loc:@gradients_1/diz_1/dense_3/BiasAdd_grad/BiasAddGrad*
T0
?
4gradients_1/diz/dense_2/LeakyRelu_grad/LeakyReluGradLeakyReluGrad<gradients_1/diz/dense_3/MatMul_grad/tuple/control_dependencydiz/dense_2/BiasAdd*
T0*'
_output_shapes
:????????? *
alpha%??L>
?
,gradients_1/diz_1/dense_3/MatMul_grad/MatMulMatMul?gradients_1/diz_1/dense_3/BiasAdd_grad/tuple/control_dependencydiz/dense_3/kernel/read*'
_output_shapes
:????????? *
transpose_a( *
transpose_b(*
T0
?
.gradients_1/diz_1/dense_3/MatMul_grad/MatMul_1MatMuldiz_1/dense_2/LeakyRelu?gradients_1/diz_1/dense_3/BiasAdd_grad/tuple/control_dependency*
_output_shapes

: *
T0*
transpose_a(*
transpose_b( 
?
6gradients_1/diz_1/dense_3/MatMul_grad/tuple/group_depsNoOp-^gradients_1/diz_1/dense_3/MatMul_grad/MatMul/^gradients_1/diz_1/dense_3/MatMul_grad/MatMul_1
?
>gradients_1/diz_1/dense_3/MatMul_grad/tuple/control_dependencyIdentity,gradients_1/diz_1/dense_3/MatMul_grad/MatMul7^gradients_1/diz_1/dense_3/MatMul_grad/tuple/group_deps*
T0*'
_output_shapes
:????????? *?
_class5
31loc:@gradients_1/diz_1/dense_3/MatMul_grad/MatMul
?
@gradients_1/diz_1/dense_3/MatMul_grad/tuple/control_dependency_1Identity.gradients_1/diz_1/dense_3/MatMul_grad/MatMul_17^gradients_1/diz_1/dense_3/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/diz_1/dense_3/MatMul_grad/MatMul_1*
_output_shapes

: 
?
gradients_1/AddNAddN?gradients_1/diz/dense_3/BiasAdd_grad/tuple/control_dependency_1Agradients_1/diz_1/dense_3/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:*
N*C
_class9
75loc:@gradients_1/diz/dense_3/BiasAdd_grad/BiasAddGrad
?
0gradients_1/diz/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients_1/diz/dense_2/LeakyRelu_grad/LeakyReluGrad*
_output_shapes
: *
T0*
data_formatNHWC
?
5gradients_1/diz/dense_2/BiasAdd_grad/tuple/group_depsNoOp1^gradients_1/diz/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_1/diz/dense_2/LeakyRelu_grad/LeakyReluGrad
?
=gradients_1/diz/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity4gradients_1/diz/dense_2/LeakyRelu_grad/LeakyReluGrad6^gradients_1/diz/dense_2/BiasAdd_grad/tuple/group_deps*G
_class=
;9loc:@gradients_1/diz/dense_2/LeakyRelu_grad/LeakyReluGrad*
T0*'
_output_shapes
:????????? 
?
?gradients_1/diz/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients_1/diz/dense_2/BiasAdd_grad/BiasAddGrad6^gradients_1/diz/dense_2/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@gradients_1/diz/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
?
6gradients_1/diz_1/dense_2/LeakyRelu_grad/LeakyReluGradLeakyReluGrad>gradients_1/diz_1/dense_3/MatMul_grad/tuple/control_dependencydiz_1/dense_2/BiasAdd*'
_output_shapes
:????????? *
T0*
alpha%??L>
?
gradients_1/AddN_1AddN>gradients_1/diz/dense_3/MatMul_grad/tuple/control_dependency_1@gradients_1/diz_1/dense_3/MatMul_grad/tuple/control_dependency_1*
T0*?
_class5
31loc:@gradients_1/diz/dense_3/MatMul_grad/MatMul_1*
N*
_output_shapes

: 
?
*gradients_1/diz/dense_2/MatMul_grad/MatMulMatMul=gradients_1/diz/dense_2/BiasAdd_grad/tuple/control_dependencydiz/dense_2/kernel/read*
transpose_b(*'
_output_shapes
:????????? *
T0*
transpose_a( 
?
,gradients_1/diz/dense_2/MatMul_grad/MatMul_1MatMuldiz/dense_1/LeakyRelu=gradients_1/diz/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes

:  
?
4gradients_1/diz/dense_2/MatMul_grad/tuple/group_depsNoOp+^gradients_1/diz/dense_2/MatMul_grad/MatMul-^gradients_1/diz/dense_2/MatMul_grad/MatMul_1
?
<gradients_1/diz/dense_2/MatMul_grad/tuple/control_dependencyIdentity*gradients_1/diz/dense_2/MatMul_grad/MatMul5^gradients_1/diz/dense_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/diz/dense_2/MatMul_grad/MatMul*'
_output_shapes
:????????? 
?
>gradients_1/diz/dense_2/MatMul_grad/tuple/control_dependency_1Identity,gradients_1/diz/dense_2/MatMul_grad/MatMul_15^gradients_1/diz/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes

:  *?
_class5
31loc:@gradients_1/diz/dense_2/MatMul_grad/MatMul_1*
T0
?
2gradients_1/diz_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients_1/diz_1/dense_2/LeakyRelu_grad/LeakyReluGrad*
data_formatNHWC*
_output_shapes
: *
T0
?
7gradients_1/diz_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp3^gradients_1/diz_1/dense_2/BiasAdd_grad/BiasAddGrad7^gradients_1/diz_1/dense_2/LeakyRelu_grad/LeakyReluGrad
?
?gradients_1/diz_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity6gradients_1/diz_1/dense_2/LeakyRelu_grad/LeakyReluGrad8^gradients_1/diz_1/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:????????? *I
_class?
=;loc:@gradients_1/diz_1/dense_2/LeakyRelu_grad/LeakyReluGrad*
T0
?
Agradients_1/diz_1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_1/diz_1/dense_2/BiasAdd_grad/BiasAddGrad8^gradients_1/diz_1/dense_2/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@gradients_1/diz_1/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
?
4gradients_1/diz/dense_1/LeakyRelu_grad/LeakyReluGradLeakyReluGrad<gradients_1/diz/dense_2/MatMul_grad/tuple/control_dependencydiz/dense_1/BiasAdd*
alpha%??L>*
T0*'
_output_shapes
:????????? 
?
,gradients_1/diz_1/dense_2/MatMul_grad/MatMulMatMul?gradients_1/diz_1/dense_2/BiasAdd_grad/tuple/control_dependencydiz/dense_2/kernel/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:????????? 
?
.gradients_1/diz_1/dense_2/MatMul_grad/MatMul_1MatMuldiz_1/dense_1/LeakyRelu?gradients_1/diz_1/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:  *
transpose_b( *
T0
?
6gradients_1/diz_1/dense_2/MatMul_grad/tuple/group_depsNoOp-^gradients_1/diz_1/dense_2/MatMul_grad/MatMul/^gradients_1/diz_1/dense_2/MatMul_grad/MatMul_1
?
>gradients_1/diz_1/dense_2/MatMul_grad/tuple/control_dependencyIdentity,gradients_1/diz_1/dense_2/MatMul_grad/MatMul7^gradients_1/diz_1/dense_2/MatMul_grad/tuple/group_deps*?
_class5
31loc:@gradients_1/diz_1/dense_2/MatMul_grad/MatMul*'
_output_shapes
:????????? *
T0
?
@gradients_1/diz_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity.gradients_1/diz_1/dense_2/MatMul_grad/MatMul_17^gradients_1/diz_1/dense_2/MatMul_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/diz_1/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:  
?
gradients_1/AddN_2AddN?gradients_1/diz/dense_2/BiasAdd_grad/tuple/control_dependency_1Agradients_1/diz_1/dense_2/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes
: *C
_class9
75loc:@gradients_1/diz/dense_2/BiasAdd_grad/BiasAddGrad*
T0
?
0gradients_1/diz/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients_1/diz/dense_1/LeakyRelu_grad/LeakyReluGrad*
_output_shapes
: *
T0*
data_formatNHWC
?
5gradients_1/diz/dense_1/BiasAdd_grad/tuple/group_depsNoOp1^gradients_1/diz/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_1/diz/dense_1/LeakyRelu_grad/LeakyReluGrad
?
=gradients_1/diz/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity4gradients_1/diz/dense_1/LeakyRelu_grad/LeakyReluGrad6^gradients_1/diz/dense_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:????????? *G
_class=
;9loc:@gradients_1/diz/dense_1/LeakyRelu_grad/LeakyReluGrad*
T0
?
?gradients_1/diz/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients_1/diz/dense_1/BiasAdd_grad/BiasAddGrad6^gradients_1/diz/dense_1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/diz/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
?
6gradients_1/diz_1/dense_1/LeakyRelu_grad/LeakyReluGradLeakyReluGrad>gradients_1/diz_1/dense_2/MatMul_grad/tuple/control_dependencydiz_1/dense_1/BiasAdd*
alpha%??L>*
T0*'
_output_shapes
:????????? 
?
gradients_1/AddN_3AddN>gradients_1/diz/dense_2/MatMul_grad/tuple/control_dependency_1@gradients_1/diz_1/dense_2/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:  *
T0*
N*?
_class5
31loc:@gradients_1/diz/dense_2/MatMul_grad/MatMul_1
?
*gradients_1/diz/dense_1/MatMul_grad/MatMulMatMul=gradients_1/diz/dense_1/BiasAdd_grad/tuple/control_dependencydiz/dense_1/kernel/read*
transpose_b(*'
_output_shapes
:????????? *
T0*
transpose_a( 
?
,gradients_1/diz/dense_1/MatMul_grad/MatMul_1MatMuldiz/dense/LeakyRelu=gradients_1/diz/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:  
?
4gradients_1/diz/dense_1/MatMul_grad/tuple/group_depsNoOp+^gradients_1/diz/dense_1/MatMul_grad/MatMul-^gradients_1/diz/dense_1/MatMul_grad/MatMul_1
?
<gradients_1/diz/dense_1/MatMul_grad/tuple/control_dependencyIdentity*gradients_1/diz/dense_1/MatMul_grad/MatMul5^gradients_1/diz/dense_1/MatMul_grad/tuple/group_deps*
T0*'
_output_shapes
:????????? *=
_class3
1/loc:@gradients_1/diz/dense_1/MatMul_grad/MatMul
?
>gradients_1/diz/dense_1/MatMul_grad/tuple/control_dependency_1Identity,gradients_1/diz/dense_1/MatMul_grad/MatMul_15^gradients_1/diz/dense_1/MatMul_grad/tuple/group_deps*?
_class5
31loc:@gradients_1/diz/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:  *
T0
?
2gradients_1/diz_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients_1/diz_1/dense_1/LeakyRelu_grad/LeakyReluGrad*
T0*
_output_shapes
: *
data_formatNHWC
?
7gradients_1/diz_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp3^gradients_1/diz_1/dense_1/BiasAdd_grad/BiasAddGrad7^gradients_1/diz_1/dense_1/LeakyRelu_grad/LeakyReluGrad
?
?gradients_1/diz_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients_1/diz_1/dense_1/LeakyRelu_grad/LeakyReluGrad8^gradients_1/diz_1/dense_1/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@gradients_1/diz_1/dense_1/LeakyRelu_grad/LeakyReluGrad*'
_output_shapes
:????????? *
T0
?
Agradients_1/diz_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_1/diz_1/dense_1/BiasAdd_grad/BiasAddGrad8^gradients_1/diz_1/dense_1/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@gradients_1/diz_1/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
?
2gradients_1/diz/dense/LeakyRelu_grad/LeakyReluGradLeakyReluGrad<gradients_1/diz/dense_1/MatMul_grad/tuple/control_dependencydiz/dense/BiasAdd*
alpha%??L>*'
_output_shapes
:????????? *
T0
?
,gradients_1/diz_1/dense_1/MatMul_grad/MatMulMatMul?gradients_1/diz_1/dense_1/BiasAdd_grad/tuple/control_dependencydiz/dense_1/kernel/read*'
_output_shapes
:????????? *
transpose_a( *
T0*
transpose_b(
?
.gradients_1/diz_1/dense_1/MatMul_grad/MatMul_1MatMuldiz_1/dense/LeakyRelu?gradients_1/diz_1/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
_output_shapes

:  *
T0
?
6gradients_1/diz_1/dense_1/MatMul_grad/tuple/group_depsNoOp-^gradients_1/diz_1/dense_1/MatMul_grad/MatMul/^gradients_1/diz_1/dense_1/MatMul_grad/MatMul_1
?
>gradients_1/diz_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity,gradients_1/diz_1/dense_1/MatMul_grad/MatMul7^gradients_1/diz_1/dense_1/MatMul_grad/tuple/group_deps*'
_output_shapes
:????????? *?
_class5
31loc:@gradients_1/diz_1/dense_1/MatMul_grad/MatMul*
T0
?
@gradients_1/diz_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity.gradients_1/diz_1/dense_1/MatMul_grad/MatMul_17^gradients_1/diz_1/dense_1/MatMul_grad/tuple/group_deps*
T0*
_output_shapes

:  *A
_class7
53loc:@gradients_1/diz_1/dense_1/MatMul_grad/MatMul_1
?
gradients_1/AddN_4AddN?gradients_1/diz/dense_1/BiasAdd_grad/tuple/control_dependency_1Agradients_1/diz_1/dense_1/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients_1/diz/dense_1/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
: 
?
.gradients_1/diz/dense/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients_1/diz/dense/LeakyRelu_grad/LeakyReluGrad*
data_formatNHWC*
T0*
_output_shapes
: 
?
3gradients_1/diz/dense/BiasAdd_grad/tuple/group_depsNoOp/^gradients_1/diz/dense/BiasAdd_grad/BiasAddGrad3^gradients_1/diz/dense/LeakyRelu_grad/LeakyReluGrad
?
;gradients_1/diz/dense/BiasAdd_grad/tuple/control_dependencyIdentity2gradients_1/diz/dense/LeakyRelu_grad/LeakyReluGrad4^gradients_1/diz/dense/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@gradients_1/diz/dense/LeakyRelu_grad/LeakyReluGrad*
T0*'
_output_shapes
:????????? 
?
=gradients_1/diz/dense/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/diz/dense/BiasAdd_grad/BiasAddGrad4^gradients_1/diz/dense/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/diz/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
?
4gradients_1/diz_1/dense/LeakyRelu_grad/LeakyReluGradLeakyReluGrad>gradients_1/diz_1/dense_1/MatMul_grad/tuple/control_dependencydiz_1/dense/BiasAdd*
alpha%??L>*
T0*'
_output_shapes
:????????? 
?
gradients_1/AddN_5AddN>gradients_1/diz/dense_1/MatMul_grad/tuple/control_dependency_1@gradients_1/diz_1/dense_1/MatMul_grad/tuple/control_dependency_1*?
_class5
31loc:@gradients_1/diz/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:  *
N*
T0
?
(gradients_1/diz/dense/MatMul_grad/MatMulMatMul;gradients_1/diz/dense/BiasAdd_grad/tuple/control_dependencydiz/dense/kernel/read*'
_output_shapes
:?????????d*
T0*
transpose_b(*
transpose_a( 
?
*gradients_1/diz/dense/MatMul_grad/MatMul_1MatMulHencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add;gradients_1/diz/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:d *
transpose_a(*
T0*
transpose_b( 
?
2gradients_1/diz/dense/MatMul_grad/tuple/group_depsNoOp)^gradients_1/diz/dense/MatMul_grad/MatMul+^gradients_1/diz/dense/MatMul_grad/MatMul_1
?
:gradients_1/diz/dense/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/diz/dense/MatMul_grad/MatMul3^gradients_1/diz/dense/MatMul_grad/tuple/group_deps*;
_class1
/-loc:@gradients_1/diz/dense/MatMul_grad/MatMul*'
_output_shapes
:?????????d*
T0
?
<gradients_1/diz/dense/MatMul_grad/tuple/control_dependency_1Identity*gradients_1/diz/dense/MatMul_grad/MatMul_13^gradients_1/diz/dense/MatMul_grad/tuple/group_deps*=
_class3
1/loc:@gradients_1/diz/dense/MatMul_grad/MatMul_1*
_output_shapes

:d *
T0
?
0gradients_1/diz_1/dense/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients_1/diz_1/dense/LeakyRelu_grad/LeakyReluGrad*
data_formatNHWC*
_output_shapes
: *
T0
?
5gradients_1/diz_1/dense/BiasAdd_grad/tuple/group_depsNoOp1^gradients_1/diz_1/dense/BiasAdd_grad/BiasAddGrad5^gradients_1/diz_1/dense/LeakyRelu_grad/LeakyReluGrad
?
=gradients_1/diz_1/dense/BiasAdd_grad/tuple/control_dependencyIdentity4gradients_1/diz_1/dense/LeakyRelu_grad/LeakyReluGrad6^gradients_1/diz_1/dense/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:????????? *G
_class=
;9loc:@gradients_1/diz_1/dense/LeakyRelu_grad/LeakyReluGrad*
T0
?
?gradients_1/diz_1/dense/BiasAdd_grad/tuple/control_dependency_1Identity0gradients_1/diz_1/dense/BiasAdd_grad/BiasAddGrad6^gradients_1/diz_1/dense/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/diz_1/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
?
*gradients_1/diz_1/dense/MatMul_grad/MatMulMatMul=gradients_1/diz_1/dense/BiasAdd_grad/tuple/control_dependencydiz/dense/kernel/read*
transpose_a( *'
_output_shapes
:?????????d*
transpose_b(*
T0
?
,gradients_1/diz_1/dense/MatMul_grad/MatMul_1MatMul@MultivariateNormalDiag/sample/affine_linear_operator/forward/add=gradients_1/diz_1/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:d *
transpose_a(*
T0*
transpose_b( 
?
4gradients_1/diz_1/dense/MatMul_grad/tuple/group_depsNoOp+^gradients_1/diz_1/dense/MatMul_grad/MatMul-^gradients_1/diz_1/dense/MatMul_grad/MatMul_1
?
<gradients_1/diz_1/dense/MatMul_grad/tuple/control_dependencyIdentity*gradients_1/diz_1/dense/MatMul_grad/MatMul5^gradients_1/diz_1/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:?????????d*=
_class3
1/loc:@gradients_1/diz_1/dense/MatMul_grad/MatMul*
T0
?
>gradients_1/diz_1/dense/MatMul_grad/tuple/control_dependency_1Identity,gradients_1/diz_1/dense/MatMul_grad/MatMul_15^gradients_1/diz_1/dense/MatMul_grad/tuple/group_deps*?
_class5
31loc:@gradients_1/diz_1/dense/MatMul_grad/MatMul_1*
_output_shapes

:d *
T0
?
gradients_1/AddN_6AddN=gradients_1/diz/dense/BiasAdd_grad/tuple/control_dependency_1?gradients_1/diz_1/dense/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes
: *
T0*A
_class7
53loc:@gradients_1/diz/dense/BiasAdd_grad/BiasAddGrad
?
gradients_1/AddN_7AddN<gradients_1/diz/dense/MatMul_grad/tuple/control_dependency_1>gradients_1/diz_1/dense/MatMul_grad/tuple/control_dependency_1*
T0*=
_class3
1/loc:@gradients_1/diz/dense/MatMul_grad/MatMul_1*
N*
_output_shapes

:d 
?
9diz/dense/kernel/RMSProp/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"d       *#
_class
loc:@diz/dense/kernel
?
/diz/dense/kernel/RMSProp/Initializer/ones/ConstConst*
_output_shapes
: *
valueB
 *  ??*
dtype0*#
_class
loc:@diz/dense/kernel
?
)diz/dense/kernel/RMSProp/Initializer/onesFill9diz/dense/kernel/RMSProp/Initializer/ones/shape_as_tensor/diz/dense/kernel/RMSProp/Initializer/ones/Const*
T0*
_output_shapes

:d *

index_type0*#
_class
loc:@diz/dense/kernel
?
diz/dense/kernel/RMSProp
VariableV2*#
_class
loc:@diz/dense/kernel*
shared_name *
dtype0*
shape
:d *
_output_shapes

:d *
	container 
?
diz/dense/kernel/RMSProp/AssignAssigndiz/dense/kernel/RMSProp)diz/dense/kernel/RMSProp/Initializer/ones*
T0*#
_class
loc:@diz/dense/kernel*
_output_shapes

:d *
validate_shape(*
use_locking(
?
diz/dense/kernel/RMSProp/readIdentitydiz/dense/kernel/RMSProp*
T0*
_output_shapes

:d *#
_class
loc:@diz/dense/kernel
?
<diz/dense/kernel/RMSProp_1/Initializer/zeros/shape_as_tensorConst*
dtype0*#
_class
loc:@diz/dense/kernel*
valueB"d       *
_output_shapes
:
?
2diz/dense/kernel/RMSProp_1/Initializer/zeros/ConstConst*#
_class
loc:@diz/dense/kernel*
dtype0*
_output_shapes
: *
valueB
 *    
?
,diz/dense/kernel/RMSProp_1/Initializer/zerosFill<diz/dense/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor2diz/dense/kernel/RMSProp_1/Initializer/zeros/Const*
_output_shapes

:d *
T0*

index_type0*#
_class
loc:@diz/dense/kernel
?
diz/dense/kernel/RMSProp_1
VariableV2*#
_class
loc:@diz/dense/kernel*
	container *
dtype0*
shape
:d *
shared_name *
_output_shapes

:d 
?
!diz/dense/kernel/RMSProp_1/AssignAssigndiz/dense/kernel/RMSProp_1,diz/dense/kernel/RMSProp_1/Initializer/zeros*
use_locking(*#
_class
loc:@diz/dense/kernel*
validate_shape(*
_output_shapes

:d *
T0
?
diz/dense/kernel/RMSProp_1/readIdentitydiz/dense/kernel/RMSProp_1*
T0*#
_class
loc:@diz/dense/kernel*
_output_shapes

:d 
?
'diz/dense/bias/RMSProp/Initializer/onesConst*
_output_shapes
: *!
_class
loc:@diz/dense/bias*
dtype0*
valueB *  ??
?
diz/dense/bias/RMSProp
VariableV2*
	container *!
_class
loc:@diz/dense/bias*
_output_shapes
: *
shape: *
dtype0*
shared_name 
?
diz/dense/bias/RMSProp/AssignAssigndiz/dense/bias/RMSProp'diz/dense/bias/RMSProp/Initializer/ones*
use_locking(*
validate_shape(*
_output_shapes
: *!
_class
loc:@diz/dense/bias*
T0
?
diz/dense/bias/RMSProp/readIdentitydiz/dense/bias/RMSProp*!
_class
loc:@diz/dense/bias*
_output_shapes
: *
T0
?
*diz/dense/bias/RMSProp_1/Initializer/zerosConst*!
_class
loc:@diz/dense/bias*
valueB *    *
_output_shapes
: *
dtype0
?
diz/dense/bias/RMSProp_1
VariableV2*
shared_name *
_output_shapes
: *
dtype0*
shape: *!
_class
loc:@diz/dense/bias*
	container 
?
diz/dense/bias/RMSProp_1/AssignAssigndiz/dense/bias/RMSProp_1*diz/dense/bias/RMSProp_1/Initializer/zeros*!
_class
loc:@diz/dense/bias*
_output_shapes
: *
validate_shape(*
T0*
use_locking(
?
diz/dense/bias/RMSProp_1/readIdentitydiz/dense/bias/RMSProp_1*
_output_shapes
: *!
_class
loc:@diz/dense/bias*
T0
?
;diz/dense_1/kernel/RMSProp/Initializer/ones/shape_as_tensorConst*%
_class
loc:@diz/dense_1/kernel*
valueB"        *
_output_shapes
:*
dtype0
?
1diz/dense_1/kernel/RMSProp/Initializer/ones/ConstConst*%
_class
loc:@diz/dense_1/kernel*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
+diz/dense_1/kernel/RMSProp/Initializer/onesFill;diz/dense_1/kernel/RMSProp/Initializer/ones/shape_as_tensor1diz/dense_1/kernel/RMSProp/Initializer/ones/Const*
T0*

index_type0*
_output_shapes

:  *%
_class
loc:@diz/dense_1/kernel
?
diz/dense_1/kernel/RMSProp
VariableV2*%
_class
loc:@diz/dense_1/kernel*
dtype0*
_output_shapes

:  *
shape
:  *
shared_name *
	container 
?
!diz/dense_1/kernel/RMSProp/AssignAssigndiz/dense_1/kernel/RMSProp+diz/dense_1/kernel/RMSProp/Initializer/ones*
use_locking(*
validate_shape(*%
_class
loc:@diz/dense_1/kernel*
_output_shapes

:  *
T0
?
diz/dense_1/kernel/RMSProp/readIdentitydiz/dense_1/kernel/RMSProp*
_output_shapes

:  *%
_class
loc:@diz/dense_1/kernel*
T0
?
>diz/dense_1/kernel/RMSProp_1/Initializer/zeros/shape_as_tensorConst*
valueB"        *
_output_shapes
:*%
_class
loc:@diz/dense_1/kernel*
dtype0
?
4diz/dense_1/kernel/RMSProp_1/Initializer/zeros/ConstConst*%
_class
loc:@diz/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
.diz/dense_1/kernel/RMSProp_1/Initializer/zerosFill>diz/dense_1/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor4diz/dense_1/kernel/RMSProp_1/Initializer/zeros/Const*
_output_shapes

:  *

index_type0*%
_class
loc:@diz/dense_1/kernel*
T0
?
diz/dense_1/kernel/RMSProp_1
VariableV2*
_output_shapes

:  *
shared_name *
dtype0*%
_class
loc:@diz/dense_1/kernel*
shape
:  *
	container 
?
#diz/dense_1/kernel/RMSProp_1/AssignAssigndiz/dense_1/kernel/RMSProp_1.diz/dense_1/kernel/RMSProp_1/Initializer/zeros*
use_locking(*
validate_shape(*
_output_shapes

:  *%
_class
loc:@diz/dense_1/kernel*
T0
?
!diz/dense_1/kernel/RMSProp_1/readIdentitydiz/dense_1/kernel/RMSProp_1*
_output_shapes

:  *%
_class
loc:@diz/dense_1/kernel*
T0
?
)diz/dense_1/bias/RMSProp/Initializer/onesConst*
dtype0*#
_class
loc:@diz/dense_1/bias*
_output_shapes
: *
valueB *  ??
?
diz/dense_1/bias/RMSProp
VariableV2*
	container *#
_class
loc:@diz/dense_1/bias*
dtype0*
shared_name *
shape: *
_output_shapes
: 
?
diz/dense_1/bias/RMSProp/AssignAssigndiz/dense_1/bias/RMSProp)diz/dense_1/bias/RMSProp/Initializer/ones*
use_locking(*
_output_shapes
: *
validate_shape(*#
_class
loc:@diz/dense_1/bias*
T0
?
diz/dense_1/bias/RMSProp/readIdentitydiz/dense_1/bias/RMSProp*
_output_shapes
: *
T0*#
_class
loc:@diz/dense_1/bias
?
,diz/dense_1/bias/RMSProp_1/Initializer/zerosConst*#
_class
loc:@diz/dense_1/bias*
valueB *    *
dtype0*
_output_shapes
: 
?
diz/dense_1/bias/RMSProp_1
VariableV2*
_output_shapes
: *
	container *
shape: *#
_class
loc:@diz/dense_1/bias*
shared_name *
dtype0
?
!diz/dense_1/bias/RMSProp_1/AssignAssigndiz/dense_1/bias/RMSProp_1,diz/dense_1/bias/RMSProp_1/Initializer/zeros*
T0*
_output_shapes
: *#
_class
loc:@diz/dense_1/bias*
use_locking(*
validate_shape(
?
diz/dense_1/bias/RMSProp_1/readIdentitydiz/dense_1/bias/RMSProp_1*#
_class
loc:@diz/dense_1/bias*
_output_shapes
: *
T0
?
;diz/dense_2/kernel/RMSProp/Initializer/ones/shape_as_tensorConst*
valueB"        *
_output_shapes
:*%
_class
loc:@diz/dense_2/kernel*
dtype0
?
1diz/dense_2/kernel/RMSProp/Initializer/ones/ConstConst*
valueB
 *  ??*
_output_shapes
: *%
_class
loc:@diz/dense_2/kernel*
dtype0
?
+diz/dense_2/kernel/RMSProp/Initializer/onesFill;diz/dense_2/kernel/RMSProp/Initializer/ones/shape_as_tensor1diz/dense_2/kernel/RMSProp/Initializer/ones/Const*
T0*
_output_shapes

:  *

index_type0*%
_class
loc:@diz/dense_2/kernel
?
diz/dense_2/kernel/RMSProp
VariableV2*
dtype0*
shared_name *%
_class
loc:@diz/dense_2/kernel*
	container *
shape
:  *
_output_shapes

:  
?
!diz/dense_2/kernel/RMSProp/AssignAssigndiz/dense_2/kernel/RMSProp+diz/dense_2/kernel/RMSProp/Initializer/ones*
_output_shapes

:  *
use_locking(*%
_class
loc:@diz/dense_2/kernel*
validate_shape(*
T0
?
diz/dense_2/kernel/RMSProp/readIdentitydiz/dense_2/kernel/RMSProp*
T0*
_output_shapes

:  *%
_class
loc:@diz/dense_2/kernel
?
>diz/dense_2/kernel/RMSProp_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
_class
loc:@diz/dense_2/kernel*
valueB"        
?
4diz/dense_2/kernel/RMSProp_1/Initializer/zeros/ConstConst*
_output_shapes
: *%
_class
loc:@diz/dense_2/kernel*
valueB
 *    *
dtype0
?
.diz/dense_2/kernel/RMSProp_1/Initializer/zerosFill>diz/dense_2/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor4diz/dense_2/kernel/RMSProp_1/Initializer/zeros/Const*
T0*

index_type0*%
_class
loc:@diz/dense_2/kernel*
_output_shapes

:  
?
diz/dense_2/kernel/RMSProp_1
VariableV2*%
_class
loc:@diz/dense_2/kernel*
_output_shapes

:  *
shape
:  *
	container *
dtype0*
shared_name 
?
#diz/dense_2/kernel/RMSProp_1/AssignAssigndiz/dense_2/kernel/RMSProp_1.diz/dense_2/kernel/RMSProp_1/Initializer/zeros*
_output_shapes

:  *
use_locking(*%
_class
loc:@diz/dense_2/kernel*
validate_shape(*
T0
?
!diz/dense_2/kernel/RMSProp_1/readIdentitydiz/dense_2/kernel/RMSProp_1*%
_class
loc:@diz/dense_2/kernel*
_output_shapes

:  *
T0
?
)diz/dense_2/bias/RMSProp/Initializer/onesConst*
_output_shapes
: *#
_class
loc:@diz/dense_2/bias*
valueB *  ??*
dtype0
?
diz/dense_2/bias/RMSProp
VariableV2*
dtype0*#
_class
loc:@diz/dense_2/bias*
	container *
_output_shapes
: *
shared_name *
shape: 
?
diz/dense_2/bias/RMSProp/AssignAssigndiz/dense_2/bias/RMSProp)diz/dense_2/bias/RMSProp/Initializer/ones*
validate_shape(*
_output_shapes
: *
T0*#
_class
loc:@diz/dense_2/bias*
use_locking(
?
diz/dense_2/bias/RMSProp/readIdentitydiz/dense_2/bias/RMSProp*
T0*#
_class
loc:@diz/dense_2/bias*
_output_shapes
: 
?
,diz/dense_2/bias/RMSProp_1/Initializer/zerosConst*
dtype0*
valueB *    *#
_class
loc:@diz/dense_2/bias*
_output_shapes
: 
?
diz/dense_2/bias/RMSProp_1
VariableV2*
_output_shapes
: *#
_class
loc:@diz/dense_2/bias*
dtype0*
shared_name *
	container *
shape: 
?
!diz/dense_2/bias/RMSProp_1/AssignAssigndiz/dense_2/bias/RMSProp_1,diz/dense_2/bias/RMSProp_1/Initializer/zeros*
validate_shape(*
T0*#
_class
loc:@diz/dense_2/bias*
_output_shapes
: *
use_locking(
?
diz/dense_2/bias/RMSProp_1/readIdentitydiz/dense_2/bias/RMSProp_1*
_output_shapes
: *#
_class
loc:@diz/dense_2/bias*
T0
?
+diz/dense_3/kernel/RMSProp/Initializer/onesConst*
_output_shapes

: *%
_class
loc:@diz/dense_3/kernel*
valueB *  ??*
dtype0
?
diz/dense_3/kernel/RMSProp
VariableV2*
	container *
_output_shapes

: *%
_class
loc:@diz/dense_3/kernel*
dtype0*
shared_name *
shape
: 
?
!diz/dense_3/kernel/RMSProp/AssignAssigndiz/dense_3/kernel/RMSProp+diz/dense_3/kernel/RMSProp/Initializer/ones*
use_locking(*
T0*
_output_shapes

: *%
_class
loc:@diz/dense_3/kernel*
validate_shape(
?
diz/dense_3/kernel/RMSProp/readIdentitydiz/dense_3/kernel/RMSProp*
T0*
_output_shapes

: *%
_class
loc:@diz/dense_3/kernel
?
.diz/dense_3/kernel/RMSProp_1/Initializer/zerosConst*%
_class
loc:@diz/dense_3/kernel*
_output_shapes

: *
dtype0*
valueB *    
?
diz/dense_3/kernel/RMSProp_1
VariableV2*
shared_name *
	container *
_output_shapes

: *
dtype0*
shape
: *%
_class
loc:@diz/dense_3/kernel
?
#diz/dense_3/kernel/RMSProp_1/AssignAssigndiz/dense_3/kernel/RMSProp_1.diz/dense_3/kernel/RMSProp_1/Initializer/zeros*
_output_shapes

: *
T0*%
_class
loc:@diz/dense_3/kernel*
use_locking(*
validate_shape(
?
!diz/dense_3/kernel/RMSProp_1/readIdentitydiz/dense_3/kernel/RMSProp_1*%
_class
loc:@diz/dense_3/kernel*
T0*
_output_shapes

: 
?
)diz/dense_3/bias/RMSProp/Initializer/onesConst*#
_class
loc:@diz/dense_3/bias*
valueB*  ??*
dtype0*
_output_shapes
:
?
diz/dense_3/bias/RMSProp
VariableV2*#
_class
loc:@diz/dense_3/bias*
dtype0*
	container *
shape:*
_output_shapes
:*
shared_name 
?
diz/dense_3/bias/RMSProp/AssignAssigndiz/dense_3/bias/RMSProp)diz/dense_3/bias/RMSProp/Initializer/ones*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*#
_class
loc:@diz/dense_3/bias
?
diz/dense_3/bias/RMSProp/readIdentitydiz/dense_3/bias/RMSProp*
_output_shapes
:*#
_class
loc:@diz/dense_3/bias*
T0
?
,diz/dense_3/bias/RMSProp_1/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*#
_class
loc:@diz/dense_3/bias
?
diz/dense_3/bias/RMSProp_1
VariableV2*
_output_shapes
:*#
_class
loc:@diz/dense_3/bias*
shape:*
	container *
dtype0*
shared_name 
?
!diz/dense_3/bias/RMSProp_1/AssignAssigndiz/dense_3/bias/RMSProp_1,diz/dense_3/bias/RMSProp_1/Initializer/zeros*
validate_shape(*#
_class
loc:@diz/dense_3/bias*
use_locking(*
T0*
_output_shapes
:
?
diz/dense_3/bias/RMSProp_1/readIdentitydiz/dense_3/bias/RMSProp_1*
T0*#
_class
loc:@diz/dense_3/bias*
_output_shapes
:
Z
RMSProp/learning_rateConst*
_output_shapes
: *
valueB
 *??8*
dtype0
R
RMSProp/decayConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
RMSProp/momentumConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
RMSProp/epsilonConst*
dtype0*
valueB
 *???.*
_output_shapes
: 
?
,RMSProp/update_diz/dense/kernel/ApplyRMSPropApplyRMSPropdiz/dense/kerneldiz/dense/kernel/RMSPropdiz/dense/kernel/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilongradients_1/AddN_7*
T0*#
_class
loc:@diz/dense/kernel*
_output_shapes

:d *
use_locking( 
?
*RMSProp/update_diz/dense/bias/ApplyRMSPropApplyRMSPropdiz/dense/biasdiz/dense/bias/RMSPropdiz/dense/bias/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilongradients_1/AddN_6*
T0*
use_locking( *!
_class
loc:@diz/dense/bias*
_output_shapes
: 
?
.RMSProp/update_diz/dense_1/kernel/ApplyRMSPropApplyRMSPropdiz/dense_1/kerneldiz/dense_1/kernel/RMSPropdiz/dense_1/kernel/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilongradients_1/AddN_5*%
_class
loc:@diz/dense_1/kernel*
T0*
_output_shapes

:  *
use_locking( 
?
,RMSProp/update_diz/dense_1/bias/ApplyRMSPropApplyRMSPropdiz/dense_1/biasdiz/dense_1/bias/RMSPropdiz/dense_1/bias/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilongradients_1/AddN_4*
_output_shapes
: *#
_class
loc:@diz/dense_1/bias*
T0*
use_locking( 
?
.RMSProp/update_diz/dense_2/kernel/ApplyRMSPropApplyRMSPropdiz/dense_2/kerneldiz/dense_2/kernel/RMSPropdiz/dense_2/kernel/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilongradients_1/AddN_3*
T0*
_output_shapes

:  *%
_class
loc:@diz/dense_2/kernel*
use_locking( 
?
,RMSProp/update_diz/dense_2/bias/ApplyRMSPropApplyRMSPropdiz/dense_2/biasdiz/dense_2/bias/RMSPropdiz/dense_2/bias/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilongradients_1/AddN_2*
use_locking( *
_output_shapes
: *
T0*#
_class
loc:@diz/dense_2/bias
?
.RMSProp/update_diz/dense_3/kernel/ApplyRMSPropApplyRMSPropdiz/dense_3/kerneldiz/dense_3/kernel/RMSPropdiz/dense_3/kernel/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilongradients_1/AddN_1*
_output_shapes

: *%
_class
loc:@diz/dense_3/kernel*
use_locking( *
T0
?
,RMSProp/update_diz/dense_3/bias/ApplyRMSPropApplyRMSPropdiz/dense_3/biasdiz/dense_3/bias/RMSPropdiz/dense_3/bias/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilongradients_1/AddN*
use_locking( *
_output_shapes
:*
T0*#
_class
loc:@diz/dense_3/bias
?
RMSPropNoOp+^RMSProp/update_diz/dense/bias/ApplyRMSProp-^RMSProp/update_diz/dense/kernel/ApplyRMSProp-^RMSProp/update_diz/dense_1/bias/ApplyRMSProp/^RMSProp/update_diz/dense_1/kernel/ApplyRMSProp-^RMSProp/update_diz/dense_2/bias/ApplyRMSProp/^RMSProp/update_diz/dense_2/kernel/ApplyRMSProp-^RMSProp/update_diz/dense_3/bias/ApplyRMSProp/^RMSProp/update_diz/dense_3/kernel/ApplyRMSProp
T
gradients_2/ShapeConst*
_output_shapes
: *
valueB *
dtype0
Z
gradients_2/grad_ys_0Const*
valueB
 *  ??*
_output_shapes
: *
dtype0
u
gradients_2/FillFillgradients_2/Shapegradients_2/grad_ys_0*

index_type0*
_output_shapes
: *
T0
B
'gradients_2/add_6_grad/tuple/group_depsNoOp^gradients_2/Fill
?
/gradients_2/add_6_grad/tuple/control_dependencyIdentitygradients_2/Fill(^gradients_2/add_6_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_2/Fill*
_output_shapes
: 
?
1gradients_2/add_6_grad/tuple/control_dependency_1Identitygradients_2/Fill(^gradients_2/add_6_grad/tuple/group_deps*
T0*
_output_shapes
: *#
_class
loc:@gradients_2/Fill
v
%gradients_2/Mean_4_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
?
gradients_2/Mean_4_grad/ReshapeReshape/gradients_2/add_6_grad/tuple/control_dependency%gradients_2/Mean_4_grad/Reshape/shape*
_output_shapes

:*
Tshape0*
T0
b
gradients_2/Mean_4_grad/ShapeShapeNeg_4*
T0*
out_type0*
_output_shapes
:
?
gradients_2/Mean_4_grad/TileTilegradients_2/Mean_4_grad/Reshapegradients_2/Mean_4_grad/Shape*

Tmultiples0*'
_output_shapes
:?????????*
T0
d
gradients_2/Mean_4_grad/Shape_1ShapeNeg_4*
out_type0*
T0*
_output_shapes
:
b
gradients_2/Mean_4_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
g
gradients_2/Mean_4_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
?
gradients_2/Mean_4_grad/ProdProdgradients_2/Mean_4_grad/Shape_1gradients_2/Mean_4_grad/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
i
gradients_2/Mean_4_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
?
gradients_2/Mean_4_grad/Prod_1Prodgradients_2/Mean_4_grad/Shape_2gradients_2/Mean_4_grad/Const_1*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
c
!gradients_2/Mean_4_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
?
gradients_2/Mean_4_grad/MaximumMaximumgradients_2/Mean_4_grad/Prod_1!gradients_2/Mean_4_grad/Maximum/y*
_output_shapes
: *
T0
?
 gradients_2/Mean_4_grad/floordivFloorDivgradients_2/Mean_4_grad/Prodgradients_2/Mean_4_grad/Maximum*
_output_shapes
: *
T0
?
gradients_2/Mean_4_grad/CastCast gradients_2/Mean_4_grad/floordiv*

SrcT0*
_output_shapes
: *
Truncate( *

DstT0
?
gradients_2/Mean_4_grad/truedivRealDivgradients_2/Mean_4_grad/Tilegradients_2/Mean_4_grad/Cast*
T0*'
_output_shapes
:?????????
v
%gradients_2/Mean_5_grad/Reshape/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
?
gradients_2/Mean_5_grad/ReshapeReshape1gradients_2/add_6_grad/tuple/control_dependency_1%gradients_2/Mean_5_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
b
gradients_2/Mean_5_grad/ShapeShapeNeg_5*
out_type0*
T0*
_output_shapes
:
?
gradients_2/Mean_5_grad/TileTilegradients_2/Mean_5_grad/Reshapegradients_2/Mean_5_grad/Shape*

Tmultiples0*'
_output_shapes
:?????????*
T0
d
gradients_2/Mean_5_grad/Shape_1ShapeNeg_5*
_output_shapes
:*
out_type0*
T0
b
gradients_2/Mean_5_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
g
gradients_2/Mean_5_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
?
gradients_2/Mean_5_grad/ProdProdgradients_2/Mean_5_grad/Shape_1gradients_2/Mean_5_grad/Const*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
i
gradients_2/Mean_5_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
gradients_2/Mean_5_grad/Prod_1Prodgradients_2/Mean_5_grad/Shape_2gradients_2/Mean_5_grad/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
c
!gradients_2/Mean_5_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
?
gradients_2/Mean_5_grad/MaximumMaximumgradients_2/Mean_5_grad/Prod_1!gradients_2/Mean_5_grad/Maximum/y*
_output_shapes
: *
T0
?
 gradients_2/Mean_5_grad/floordivFloorDivgradients_2/Mean_5_grad/Prodgradients_2/Mean_5_grad/Maximum*
T0*
_output_shapes
: 
?
gradients_2/Mean_5_grad/CastCast gradients_2/Mean_5_grad/floordiv*
Truncate( *

DstT0*

SrcT0*
_output_shapes
: 
?
gradients_2/Mean_5_grad/truedivRealDivgradients_2/Mean_5_grad/Tilegradients_2/Mean_5_grad/Cast*
T0*'
_output_shapes
:?????????
t
gradients_2/Neg_4_grad/NegNeggradients_2/Mean_4_grad/truediv*'
_output_shapes
:?????????*
T0
t
gradients_2/Neg_5_grad/NegNeggradients_2/Mean_5_grad/truediv*
T0*'
_output_shapes
:?????????
?
!gradients_2/Log_2_grad/Reciprocal
Reciprocaladd_4^gradients_2/Neg_4_grad/Neg*'
_output_shapes
:?????????*
T0
?
gradients_2/Log_2_grad/mulMulgradients_2/Neg_4_grad/Neg!gradients_2/Log_2_grad/Reciprocal*
T0*'
_output_shapes
:?????????
?
!gradients_2/Log_3_grad/Reciprocal
Reciprocaladd_5^gradients_2/Neg_5_grad/Neg*
T0*'
_output_shapes
:?????????
?
gradients_2/Log_3_grad/mulMulgradients_2/Neg_5_grad/Neg!gradients_2/Log_3_grad/Reciprocal*'
_output_shapes
:?????????*
T0
e
gradients_2/add_4_grad/ShapeShape	Sigmoid_2*
_output_shapes
:*
out_type0*
T0
a
gradients_2/add_4_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
?
,gradients_2/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/add_4_grad/Shapegradients_2/add_4_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_2/add_4_grad/SumSumgradients_2/Log_2_grad/mul,gradients_2/add_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
?
gradients_2/add_4_grad/ReshapeReshapegradients_2/add_4_grad/Sumgradients_2/add_4_grad/Shape*
T0*'
_output_shapes
:?????????*
Tshape0
?
gradients_2/add_4_grad/Sum_1Sumgradients_2/Log_2_grad/mul.gradients_2/add_4_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
?
 gradients_2/add_4_grad/Reshape_1Reshapegradients_2/add_4_grad/Sum_1gradients_2/add_4_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
s
'gradients_2/add_4_grad/tuple/group_depsNoOp^gradients_2/add_4_grad/Reshape!^gradients_2/add_4_grad/Reshape_1
?
/gradients_2/add_4_grad/tuple/control_dependencyIdentitygradients_2/add_4_grad/Reshape(^gradients_2/add_4_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*1
_class'
%#loc:@gradients_2/add_4_grad/Reshape
?
1gradients_2/add_4_grad/tuple/control_dependency_1Identity gradients_2/add_4_grad/Reshape_1(^gradients_2/add_4_grad/tuple/group_deps*3
_class)
'%loc:@gradients_2/add_4_grad/Reshape_1*
_output_shapes
: *
T0
a
gradients_2/add_5_grad/ShapeShapesub_1*
out_type0*
_output_shapes
:*
T0
a
gradients_2/add_5_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
?
,gradients_2/add_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/add_5_grad/Shapegradients_2/add_5_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_2/add_5_grad/SumSumgradients_2/Log_3_grad/mul,gradients_2/add_5_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
gradients_2/add_5_grad/ReshapeReshapegradients_2/add_5_grad/Sumgradients_2/add_5_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients_2/add_5_grad/Sum_1Sumgradients_2/Log_3_grad/mul.gradients_2/add_5_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
?
 gradients_2/add_5_grad/Reshape_1Reshapegradients_2/add_5_grad/Sum_1gradients_2/add_5_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
s
'gradients_2/add_5_grad/tuple/group_depsNoOp^gradients_2/add_5_grad/Reshape!^gradients_2/add_5_grad/Reshape_1
?
/gradients_2/add_5_grad/tuple/control_dependencyIdentitygradients_2/add_5_grad/Reshape(^gradients_2/add_5_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_2/add_5_grad/Reshape*'
_output_shapes
:?????????
?
1gradients_2/add_5_grad/tuple/control_dependency_1Identity gradients_2/add_5_grad/Reshape_1(^gradients_2/add_5_grad/tuple/group_deps*
_output_shapes
: *3
_class)
'%loc:@gradients_2/add_5_grad/Reshape_1*
T0
?
&gradients_2/Sigmoid_2_grad/SigmoidGradSigmoidGrad	Sigmoid_2/gradients_2/add_4_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????
_
gradients_2/sub_1_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
g
gradients_2/sub_1_grad/Shape_1Shape	Sigmoid_3*
_output_shapes
:*
T0*
out_type0
?
,gradients_2/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/sub_1_grad/Shapegradients_2/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_2/sub_1_grad/SumSum/gradients_2/add_5_grad/tuple/control_dependency,gradients_2/sub_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
gradients_2/sub_1_grad/ReshapeReshapegradients_2/sub_1_grad/Sumgradients_2/sub_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
?
gradients_2/sub_1_grad/Sum_1Sum/gradients_2/add_5_grad/tuple/control_dependency.gradients_2/sub_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
b
gradients_2/sub_1_grad/NegNeggradients_2/sub_1_grad/Sum_1*
T0*
_output_shapes
:
?
 gradients_2/sub_1_grad/Reshape_1Reshapegradients_2/sub_1_grad/Neggradients_2/sub_1_grad/Shape_1*
T0*'
_output_shapes
:?????????*
Tshape0
s
'gradients_2/sub_1_grad/tuple/group_depsNoOp^gradients_2/sub_1_grad/Reshape!^gradients_2/sub_1_grad/Reshape_1
?
/gradients_2/sub_1_grad/tuple/control_dependencyIdentitygradients_2/sub_1_grad/Reshape(^gradients_2/sub_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients_2/sub_1_grad/Reshape*
_output_shapes
: *
T0
?
1gradients_2/sub_1_grad/tuple/control_dependency_1Identity gradients_2/sub_1_grad/Reshape_1(^gradients_2/sub_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_2/sub_1_grad/Reshape_1*'
_output_shapes
:?????????*
T0
?
2gradients_2/disc0/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad&gradients_2/Sigmoid_2_grad/SigmoidGrad*
data_formatNHWC*
T0*
_output_shapes
:
?
7gradients_2/disc0/dense_1/BiasAdd_grad/tuple/group_depsNoOp'^gradients_2/Sigmoid_2_grad/SigmoidGrad3^gradients_2/disc0/dense_1/BiasAdd_grad/BiasAddGrad
?
?gradients_2/disc0/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity&gradients_2/Sigmoid_2_grad/SigmoidGrad8^gradients_2/disc0/dense_1/BiasAdd_grad/tuple/group_deps*9
_class/
-+loc:@gradients_2/Sigmoid_2_grad/SigmoidGrad*
T0*'
_output_shapes
:?????????
?
Agradients_2/disc0/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_2/disc0/dense_1/BiasAdd_grad/BiasAddGrad8^gradients_2/disc0/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*E
_class;
97loc:@gradients_2/disc0/dense_1/BiasAdd_grad/BiasAddGrad
?
&gradients_2/Sigmoid_3_grad/SigmoidGradSigmoidGrad	Sigmoid_31gradients_2/sub_1_grad/tuple/control_dependency_1*'
_output_shapes
:?????????*
T0
?
,gradients_2/disc0/dense_1/MatMul_grad/MatMulMatMul?gradients_2/disc0/dense_1/BiasAdd_grad/tuple/control_dependencydisc0/dense_1/kernel/read*
transpose_b(*(
_output_shapes
:??????????*
transpose_a( *
T0
?
.gradients_2/disc0/dense_1/MatMul_grad/MatMul_1MatMuldisc0/dense/Relu?gradients_2/disc0/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	?
?
6gradients_2/disc0/dense_1/MatMul_grad/tuple/group_depsNoOp-^gradients_2/disc0/dense_1/MatMul_grad/MatMul/^gradients_2/disc0/dense_1/MatMul_grad/MatMul_1
?
>gradients_2/disc0/dense_1/MatMul_grad/tuple/control_dependencyIdentity,gradients_2/disc0/dense_1/MatMul_grad/MatMul7^gradients_2/disc0/dense_1/MatMul_grad/tuple/group_deps*?
_class5
31loc:@gradients_2/disc0/dense_1/MatMul_grad/MatMul*
T0*(
_output_shapes
:??????????
?
@gradients_2/disc0/dense_1/MatMul_grad/tuple/control_dependency_1Identity.gradients_2/disc0/dense_1/MatMul_grad/MatMul_17^gradients_2/disc0/dense_1/MatMul_grad/tuple/group_deps*A
_class7
53loc:@gradients_2/disc0/dense_1/MatMul_grad/MatMul_1*
_output_shapes
:	?*
T0
?
4gradients_2/disc0_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad&gradients_2/Sigmoid_3_grad/SigmoidGrad*
T0*
_output_shapes
:*
data_formatNHWC
?
9gradients_2/disc0_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp'^gradients_2/Sigmoid_3_grad/SigmoidGrad5^gradients_2/disc0_1/dense_1/BiasAdd_grad/BiasAddGrad
?
Agradients_2/disc0_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity&gradients_2/Sigmoid_3_grad/SigmoidGrad:^gradients_2/disc0_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*9
_class/
-+loc:@gradients_2/Sigmoid_3_grad/SigmoidGrad
?
Cgradients_2/disc0_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_2/disc0_1/dense_1/BiasAdd_grad/BiasAddGrad:^gradients_2/disc0_1/dense_1/BiasAdd_grad/tuple/group_deps*G
_class=
;9loc:@gradients_2/disc0_1/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
?
*gradients_2/disc0/dense/Relu_grad/ReluGradReluGrad>gradients_2/disc0/dense_1/MatMul_grad/tuple/control_dependencydisc0/dense/Relu*(
_output_shapes
:??????????*
T0
?
.gradients_2/disc0_1/dense_1/MatMul_grad/MatMulMatMulAgradients_2/disc0_1/dense_1/BiasAdd_grad/tuple/control_dependencydisc0/dense_1/kernel/read*
T0*
transpose_a( *
transpose_b(*(
_output_shapes
:??????????
?
0gradients_2/disc0_1/dense_1/MatMul_grad/MatMul_1MatMuldisc0_1/dense/ReluAgradients_2/disc0_1/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes
:	?*
transpose_b( *
T0
?
8gradients_2/disc0_1/dense_1/MatMul_grad/tuple/group_depsNoOp/^gradients_2/disc0_1/dense_1/MatMul_grad/MatMul1^gradients_2/disc0_1/dense_1/MatMul_grad/MatMul_1
?
@gradients_2/disc0_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity.gradients_2/disc0_1/dense_1/MatMul_grad/MatMul9^gradients_2/disc0_1/dense_1/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:??????????*A
_class7
53loc:@gradients_2/disc0_1/dense_1/MatMul_grad/MatMul
?
Bgradients_2/disc0_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity0gradients_2/disc0_1/dense_1/MatMul_grad/MatMul_19^gradients_2/disc0_1/dense_1/MatMul_grad/tuple/group_deps*C
_class9
75loc:@gradients_2/disc0_1/dense_1/MatMul_grad/MatMul_1*
_output_shapes
:	?*
T0
?
gradients_2/AddNAddNAgradients_2/disc0/dense_1/BiasAdd_grad/tuple/control_dependency_1Cgradients_2/disc0_1/dense_1/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:*E
_class;
97loc:@gradients_2/disc0/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
N
?
0gradients_2/disc0/dense/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients_2/disc0/dense/Relu_grad/ReluGrad*
_output_shapes	
:?*
T0*
data_formatNHWC
?
5gradients_2/disc0/dense/BiasAdd_grad/tuple/group_depsNoOp1^gradients_2/disc0/dense/BiasAdd_grad/BiasAddGrad+^gradients_2/disc0/dense/Relu_grad/ReluGrad
?
=gradients_2/disc0/dense/BiasAdd_grad/tuple/control_dependencyIdentity*gradients_2/disc0/dense/Relu_grad/ReluGrad6^gradients_2/disc0/dense/BiasAdd_grad/tuple/group_deps*
T0*(
_output_shapes
:??????????*=
_class3
1/loc:@gradients_2/disc0/dense/Relu_grad/ReluGrad
?
?gradients_2/disc0/dense/BiasAdd_grad/tuple/control_dependency_1Identity0gradients_2/disc0/dense/BiasAdd_grad/BiasAddGrad6^gradients_2/disc0/dense/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes	
:?*C
_class9
75loc:@gradients_2/disc0/dense/BiasAdd_grad/BiasAddGrad
?
,gradients_2/disc0_1/dense/Relu_grad/ReluGradReluGrad@gradients_2/disc0_1/dense_1/MatMul_grad/tuple/control_dependencydisc0_1/dense/Relu*
T0*(
_output_shapes
:??????????
?
gradients_2/AddN_1AddN@gradients_2/disc0/dense_1/MatMul_grad/tuple/control_dependency_1Bgradients_2/disc0_1/dense_1/MatMul_grad/tuple/control_dependency_1*A
_class7
53loc:@gradients_2/disc0/dense_1/MatMul_grad/MatMul_1*
_output_shapes
:	?*
N*
T0
?
*gradients_2/disc0/dense/MatMul_grad/MatMulMatMul=gradients_2/disc0/dense/BiasAdd_grad/tuple/control_dependencydisc0/dense/kernel/read*
T0*
transpose_b(*(
_output_shapes
:??????????*
transpose_a( 
?
,gradients_2/disc0/dense/MatMul_grad/MatMul_1MatMuldisc0/flatten/Reshape=gradients_2/disc0/dense/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
??*
transpose_b( *
transpose_a(*
T0
?
4gradients_2/disc0/dense/MatMul_grad/tuple/group_depsNoOp+^gradients_2/disc0/dense/MatMul_grad/MatMul-^gradients_2/disc0/dense/MatMul_grad/MatMul_1
?
<gradients_2/disc0/dense/MatMul_grad/tuple/control_dependencyIdentity*gradients_2/disc0/dense/MatMul_grad/MatMul5^gradients_2/disc0/dense/MatMul_grad/tuple/group_deps*=
_class3
1/loc:@gradients_2/disc0/dense/MatMul_grad/MatMul*
T0*(
_output_shapes
:??????????
?
>gradients_2/disc0/dense/MatMul_grad/tuple/control_dependency_1Identity,gradients_2/disc0/dense/MatMul_grad/MatMul_15^gradients_2/disc0/dense/MatMul_grad/tuple/group_deps* 
_output_shapes
:
??*?
_class5
31loc:@gradients_2/disc0/dense/MatMul_grad/MatMul_1*
T0
?
2gradients_2/disc0_1/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_2/disc0_1/dense/Relu_grad/ReluGrad*
_output_shapes	
:?*
data_formatNHWC*
T0
?
7gradients_2/disc0_1/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients_2/disc0_1/dense/BiasAdd_grad/BiasAddGrad-^gradients_2/disc0_1/dense/Relu_grad/ReluGrad
?
?gradients_2/disc0_1/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients_2/disc0_1/dense/Relu_grad/ReluGrad8^gradients_2/disc0_1/dense/BiasAdd_grad/tuple/group_deps*?
_class5
31loc:@gradients_2/disc0_1/dense/Relu_grad/ReluGrad*(
_output_shapes
:??????????*
T0
?
Agradients_2/disc0_1/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_2/disc0_1/dense/BiasAdd_grad/BiasAddGrad8^gradients_2/disc0_1/dense/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@gradients_2/disc0_1/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?*
T0
{
,gradients_2/disc0/flatten/Reshape_grad/ShapeShapedisc0/MaxPool_1*
_output_shapes
:*
out_type0*
T0
?
.gradients_2/disc0/flatten/Reshape_grad/ReshapeReshape<gradients_2/disc0/dense/MatMul_grad/tuple/control_dependency,gradients_2/disc0/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:?????????2
?
,gradients_2/disc0_1/dense/MatMul_grad/MatMulMatMul?gradients_2/disc0_1/dense/BiasAdd_grad/tuple/control_dependencydisc0/dense/kernel/read*
T0*(
_output_shapes
:??????????*
transpose_b(*
transpose_a( 
?
.gradients_2/disc0_1/dense/MatMul_grad/MatMul_1MatMuldisc0_1/flatten/Reshape?gradients_2/disc0_1/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( * 
_output_shapes
:
??*
T0
?
6gradients_2/disc0_1/dense/MatMul_grad/tuple/group_depsNoOp-^gradients_2/disc0_1/dense/MatMul_grad/MatMul/^gradients_2/disc0_1/dense/MatMul_grad/MatMul_1
?
>gradients_2/disc0_1/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients_2/disc0_1/dense/MatMul_grad/MatMul7^gradients_2/disc0_1/dense/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:??????????*?
_class5
31loc:@gradients_2/disc0_1/dense/MatMul_grad/MatMul
?
@gradients_2/disc0_1/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients_2/disc0_1/dense/MatMul_grad/MatMul_17^gradients_2/disc0_1/dense/MatMul_grad/tuple/group_deps*
T0* 
_output_shapes
:
??*A
_class7
53loc:@gradients_2/disc0_1/dense/MatMul_grad/MatMul_1
?
gradients_2/AddN_2AddN?gradients_2/disc0/dense/BiasAdd_grad/tuple/control_dependency_1Agradients_2/disc0_1/dense/BiasAdd_grad/tuple/control_dependency_1*
N*C
_class9
75loc:@gradients_2/disc0/dense/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:?
?
,gradients_2/disc0/MaxPool_1_grad/MaxPoolGradMaxPoolGraddisc0/conv2d_1/Reludisc0/MaxPool_1.gradients_2/disc0/flatten/Reshape_grad/Reshape*/
_output_shapes
:?????????2*
ksize
*
strides
*
paddingSAME*
data_formatNHWC*
T0

.gradients_2/disc0_1/flatten/Reshape_grad/ShapeShapedisc0_1/MaxPool_1*
out_type0*
T0*
_output_shapes
:
?
0gradients_2/disc0_1/flatten/Reshape_grad/ReshapeReshape>gradients_2/disc0_1/dense/MatMul_grad/tuple/control_dependency.gradients_2/disc0_1/flatten/Reshape_grad/Shape*
T0*/
_output_shapes
:?????????2*
Tshape0
?
gradients_2/AddN_3AddN>gradients_2/disc0/dense/MatMul_grad/tuple/control_dependency_1@gradients_2/disc0_1/dense/MatMul_grad/tuple/control_dependency_1*?
_class5
31loc:@gradients_2/disc0/dense/MatMul_grad/MatMul_1* 
_output_shapes
:
??*
T0*
N
?
-gradients_2/disc0/conv2d_1/Relu_grad/ReluGradReluGrad,gradients_2/disc0/MaxPool_1_grad/MaxPoolGraddisc0/conv2d_1/Relu*
T0*/
_output_shapes
:?????????2
?
.gradients_2/disc0_1/MaxPool_1_grad/MaxPoolGradMaxPoolGraddisc0_1/conv2d_1/Reludisc0_1/MaxPool_10gradients_2/disc0_1/flatten/Reshape_grad/Reshape*
data_formatNHWC*/
_output_shapes
:?????????2*
ksize
*
T0*
strides
*
paddingSAME
?
3gradients_2/disc0/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients_2/disc0/conv2d_1/Relu_grad/ReluGrad*
T0*
_output_shapes
:2*
data_formatNHWC
?
8gradients_2/disc0/conv2d_1/BiasAdd_grad/tuple/group_depsNoOp4^gradients_2/disc0/conv2d_1/BiasAdd_grad/BiasAddGrad.^gradients_2/disc0/conv2d_1/Relu_grad/ReluGrad
?
@gradients_2/disc0/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity-gradients_2/disc0/conv2d_1/Relu_grad/ReluGrad9^gradients_2/disc0/conv2d_1/BiasAdd_grad/tuple/group_deps*@
_class6
42loc:@gradients_2/disc0/conv2d_1/Relu_grad/ReluGrad*
T0*/
_output_shapes
:?????????2
?
Bgradients_2/disc0/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity3gradients_2/disc0/conv2d_1/BiasAdd_grad/BiasAddGrad9^gradients_2/disc0/conv2d_1/BiasAdd_grad/tuple/group_deps*F
_class<
:8loc:@gradients_2/disc0/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:2*
T0
?
/gradients_2/disc0_1/conv2d_1/Relu_grad/ReluGradReluGrad.gradients_2/disc0_1/MaxPool_1_grad/MaxPoolGraddisc0_1/conv2d_1/Relu*/
_output_shapes
:?????????2*
T0
?
-gradients_2/disc0/conv2d_1/Conv2D_grad/ShapeNShapeNdisc0/MaxPooldisc0/conv2d_1/kernel/read*
N*
T0* 
_output_shapes
::*
out_type0
?
:gradients_2/disc0/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput-gradients_2/disc0/conv2d_1/Conv2D_grad/ShapeNdisc0/conv2d_1/kernel/read@gradients_2/disc0/conv2d_1/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
explicit_paddings
 *
strides
*
T0*
use_cudnn_on_gpu(*
paddingVALID*
	dilations
*/
_output_shapes
:?????????
?
;gradients_2/disc0/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdisc0/MaxPool/gradients_2/disc0/conv2d_1/Conv2D_grad/ShapeN:1@gradients_2/disc0/conv2d_1/BiasAdd_grad/tuple/control_dependency*
strides
*&
_output_shapes
:2*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
use_cudnn_on_gpu(*
T0*
paddingVALID
?
7gradients_2/disc0/conv2d_1/Conv2D_grad/tuple/group_depsNoOp<^gradients_2/disc0/conv2d_1/Conv2D_grad/Conv2DBackpropFilter;^gradients_2/disc0/conv2d_1/Conv2D_grad/Conv2DBackpropInput
?
?gradients_2/disc0/conv2d_1/Conv2D_grad/tuple/control_dependencyIdentity:gradients_2/disc0/conv2d_1/Conv2D_grad/Conv2DBackpropInput8^gradients_2/disc0/conv2d_1/Conv2D_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_2/disc0/conv2d_1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????
?
Agradients_2/disc0/conv2d_1/Conv2D_grad/tuple/control_dependency_1Identity;gradients_2/disc0/conv2d_1/Conv2D_grad/Conv2DBackpropFilter8^gradients_2/disc0/conv2d_1/Conv2D_grad/tuple/group_deps*
T0*&
_output_shapes
:2*N
_classD
B@loc:@gradients_2/disc0/conv2d_1/Conv2D_grad/Conv2DBackpropFilter
?
5gradients_2/disc0_1/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad/gradients_2/disc0_1/conv2d_1/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:2
?
:gradients_2/disc0_1/conv2d_1/BiasAdd_grad/tuple/group_depsNoOp6^gradients_2/disc0_1/conv2d_1/BiasAdd_grad/BiasAddGrad0^gradients_2/disc0_1/conv2d_1/Relu_grad/ReluGrad
?
Bgradients_2/disc0_1/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity/gradients_2/disc0_1/conv2d_1/Relu_grad/ReluGrad;^gradients_2/disc0_1/conv2d_1/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:?????????2*
T0*B
_class8
64loc:@gradients_2/disc0_1/conv2d_1/Relu_grad/ReluGrad
?
Dgradients_2/disc0_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity5gradients_2/disc0_1/conv2d_1/BiasAdd_grad/BiasAddGrad;^gradients_2/disc0_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:2*
T0*H
_class>
<:loc:@gradients_2/disc0_1/conv2d_1/BiasAdd_grad/BiasAddGrad
?
*gradients_2/disc0/MaxPool_grad/MaxPoolGradMaxPoolGraddisc0/conv2d/Reludisc0/MaxPool?gradients_2/disc0/conv2d_1/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*/
_output_shapes
:?????????*
T0*
paddingSAME*
ksize

?
/gradients_2/disc0_1/conv2d_1/Conv2D_grad/ShapeNShapeNdisc0_1/MaxPooldisc0/conv2d_1/kernel/read*
out_type0*
T0* 
_output_shapes
::*
N
?
<gradients_2/disc0_1/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput/gradients_2/disc0_1/conv2d_1/Conv2D_grad/ShapeNdisc0/conv2d_1/kernel/readBgradients_2/disc0_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
	dilations
*
use_cudnn_on_gpu(*
strides
*/
_output_shapes
:?????????*
T0*
paddingVALID*
explicit_paddings
 
?
=gradients_2/disc0_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdisc0_1/MaxPool1gradients_2/disc0_1/conv2d_1/Conv2D_grad/ShapeN:1Bgradients_2/disc0_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
strides
*&
_output_shapes
:2*
use_cudnn_on_gpu(*
data_formatNHWC*
explicit_paddings
 *
	dilations
*
paddingVALID*
T0
?
9gradients_2/disc0_1/conv2d_1/Conv2D_grad/tuple/group_depsNoOp>^gradients_2/disc0_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilter=^gradients_2/disc0_1/conv2d_1/Conv2D_grad/Conv2DBackpropInput
?
Agradients_2/disc0_1/conv2d_1/Conv2D_grad/tuple/control_dependencyIdentity<gradients_2/disc0_1/conv2d_1/Conv2D_grad/Conv2DBackpropInput:^gradients_2/disc0_1/conv2d_1/Conv2D_grad/tuple/group_deps*/
_output_shapes
:?????????*
T0*O
_classE
CAloc:@gradients_2/disc0_1/conv2d_1/Conv2D_grad/Conv2DBackpropInput
?
Cgradients_2/disc0_1/conv2d_1/Conv2D_grad/tuple/control_dependency_1Identity=gradients_2/disc0_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilter:^gradients_2/disc0_1/conv2d_1/Conv2D_grad/tuple/group_deps*
T0*&
_output_shapes
:2*P
_classF
DBloc:@gradients_2/disc0_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilter
?
gradients_2/AddN_4AddNBgradients_2/disc0/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Dgradients_2/disc0_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1*F
_class<
:8loc:@gradients_2/disc0/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:2*
N
?
+gradients_2/disc0/conv2d/Relu_grad/ReluGradReluGrad*gradients_2/disc0/MaxPool_grad/MaxPoolGraddisc0/conv2d/Relu*/
_output_shapes
:?????????*
T0
?
,gradients_2/disc0_1/MaxPool_grad/MaxPoolGradMaxPoolGraddisc0_1/conv2d/Reludisc0_1/MaxPoolAgradients_2/disc0_1/conv2d_1/Conv2D_grad/tuple/control_dependency*/
_output_shapes
:?????????*
paddingSAME*
strides
*
data_formatNHWC*
ksize
*
T0
?
gradients_2/AddN_5AddNAgradients_2/disc0/conv2d_1/Conv2D_grad/tuple/control_dependency_1Cgradients_2/disc0_1/conv2d_1/Conv2D_grad/tuple/control_dependency_1*
N*
T0*&
_output_shapes
:2*N
_classD
B@loc:@gradients_2/disc0/conv2d_1/Conv2D_grad/Conv2DBackpropFilter
?
1gradients_2/disc0/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad+gradients_2/disc0/conv2d/Relu_grad/ReluGrad*
T0*
_output_shapes
:*
data_formatNHWC
?
6gradients_2/disc0/conv2d/BiasAdd_grad/tuple/group_depsNoOp2^gradients_2/disc0/conv2d/BiasAdd_grad/BiasAddGrad,^gradients_2/disc0/conv2d/Relu_grad/ReluGrad
?
>gradients_2/disc0/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity+gradients_2/disc0/conv2d/Relu_grad/ReluGrad7^gradients_2/disc0/conv2d/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:?????????*>
_class4
20loc:@gradients_2/disc0/conv2d/Relu_grad/ReluGrad*
T0
?
@gradients_2/disc0/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity1gradients_2/disc0/conv2d/BiasAdd_grad/BiasAddGrad7^gradients_2/disc0/conv2d/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*D
_class:
86loc:@gradients_2/disc0/conv2d/BiasAdd_grad/BiasAddGrad
?
-gradients_2/disc0_1/conv2d/Relu_grad/ReluGradReluGrad,gradients_2/disc0_1/MaxPool_grad/MaxPoolGraddisc0_1/conv2d/Relu*/
_output_shapes
:?????????*
T0
?
+gradients_2/disc0/conv2d/Conv2D_grad/ShapeNShapeNdisc0/Reshapedisc0/conv2d/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
8gradients_2/disc0/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput+gradients_2/disc0/conv2d/Conv2D_grad/ShapeNdisc0/conv2d/kernel/read>gradients_2/disc0/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
paddingVALID*
T0*
	dilations
*
use_cudnn_on_gpu(*
explicit_paddings
 */
_output_shapes
:?????????
?
9gradients_2/disc0/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdisc0/Reshape-gradients_2/disc0/conv2d/Conv2D_grad/ShapeN:1>gradients_2/disc0/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
paddingVALID*&
_output_shapes
:*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
T0*
	dilations

?
5gradients_2/disc0/conv2d/Conv2D_grad/tuple/group_depsNoOp:^gradients_2/disc0/conv2d/Conv2D_grad/Conv2DBackpropFilter9^gradients_2/disc0/conv2d/Conv2D_grad/Conv2DBackpropInput
?
=gradients_2/disc0/conv2d/Conv2D_grad/tuple/control_dependencyIdentity8gradients_2/disc0/conv2d/Conv2D_grad/Conv2DBackpropInput6^gradients_2/disc0/conv2d/Conv2D_grad/tuple/group_deps*
T0*/
_output_shapes
:?????????*K
_classA
?=loc:@gradients_2/disc0/conv2d/Conv2D_grad/Conv2DBackpropInput
?
?gradients_2/disc0/conv2d/Conv2D_grad/tuple/control_dependency_1Identity9gradients_2/disc0/conv2d/Conv2D_grad/Conv2DBackpropFilter6^gradients_2/disc0/conv2d/Conv2D_grad/tuple/group_deps*L
_classB
@>loc:@gradients_2/disc0/conv2d/Conv2D_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
?
3gradients_2/disc0_1/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients_2/disc0_1/conv2d/Relu_grad/ReluGrad*
_output_shapes
:*
data_formatNHWC*
T0
?
8gradients_2/disc0_1/conv2d/BiasAdd_grad/tuple/group_depsNoOp4^gradients_2/disc0_1/conv2d/BiasAdd_grad/BiasAddGrad.^gradients_2/disc0_1/conv2d/Relu_grad/ReluGrad
?
@gradients_2/disc0_1/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity-gradients_2/disc0_1/conv2d/Relu_grad/ReluGrad9^gradients_2/disc0_1/conv2d/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:?????????*
T0*@
_class6
42loc:@gradients_2/disc0_1/conv2d/Relu_grad/ReluGrad
?
Bgradients_2/disc0_1/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity3gradients_2/disc0_1/conv2d/BiasAdd_grad/BiasAddGrad9^gradients_2/disc0_1/conv2d/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*F
_class<
:8loc:@gradients_2/disc0_1/conv2d/BiasAdd_grad/BiasAddGrad
?
-gradients_2/disc0_1/conv2d/Conv2D_grad/ShapeNShapeNdisc0_1/Reshapedisc0/conv2d/kernel/read* 
_output_shapes
::*
N*
T0*
out_type0
?
:gradients_2/disc0_1/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput-gradients_2/disc0_1/conv2d/Conv2D_grad/ShapeNdisc0/conv2d/kernel/read@gradients_2/disc0_1/conv2d/BiasAdd_grad/tuple/control_dependency*
explicit_paddings
 *
T0*
	dilations
*
paddingVALID*/
_output_shapes
:?????????*
strides
*
use_cudnn_on_gpu(*
data_formatNHWC
?
;gradients_2/disc0_1/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdisc0_1/Reshape/gradients_2/disc0_1/conv2d/Conv2D_grad/ShapeN:1@gradients_2/disc0_1/conv2d/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
:*
explicit_paddings
 *
data_formatNHWC*
strides
*
	dilations
*
T0*
paddingVALID*
use_cudnn_on_gpu(
?
7gradients_2/disc0_1/conv2d/Conv2D_grad/tuple/group_depsNoOp<^gradients_2/disc0_1/conv2d/Conv2D_grad/Conv2DBackpropFilter;^gradients_2/disc0_1/conv2d/Conv2D_grad/Conv2DBackpropInput
?
?gradients_2/disc0_1/conv2d/Conv2D_grad/tuple/control_dependencyIdentity:gradients_2/disc0_1/conv2d/Conv2D_grad/Conv2DBackpropInput8^gradients_2/disc0_1/conv2d/Conv2D_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_2/disc0_1/conv2d/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????
?
Agradients_2/disc0_1/conv2d/Conv2D_grad/tuple/control_dependency_1Identity;gradients_2/disc0_1/conv2d/Conv2D_grad/Conv2DBackpropFilter8^gradients_2/disc0_1/conv2d/Conv2D_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_2/disc0_1/conv2d/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
?
gradients_2/AddN_6AddN@gradients_2/disc0/conv2d/BiasAdd_grad/tuple/control_dependency_1Bgradients_2/disc0_1/conv2d/BiasAdd_grad/tuple/control_dependency_1*
N*
T0*D
_class:
86loc:@gradients_2/disc0/conv2d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
?
gradients_2/AddN_7AddN?gradients_2/disc0/conv2d/Conv2D_grad/tuple/control_dependency_1Agradients_2/disc0_1/conv2d/Conv2D_grad/tuple/control_dependency_1*
T0*&
_output_shapes
:*
N*L
_classB
@>loc:@gradients_2/disc0/conv2d/Conv2D_grad/Conv2DBackpropFilter
?
,disc0/conv2d/kernel/RMSProp/Initializer/onesConst*&
_class
loc:@disc0/conv2d/kernel*
dtype0*&
_output_shapes
:*%
valueB*  ??
?
disc0/conv2d/kernel/RMSProp
VariableV2*
dtype0*&
_output_shapes
:*&
_class
loc:@disc0/conv2d/kernel*
	container *
shared_name *
shape:
?
"disc0/conv2d/kernel/RMSProp/AssignAssigndisc0/conv2d/kernel/RMSProp,disc0/conv2d/kernel/RMSProp/Initializer/ones*&
_output_shapes
:*&
_class
loc:@disc0/conv2d/kernel*
T0*
validate_shape(*
use_locking(
?
 disc0/conv2d/kernel/RMSProp/readIdentitydisc0/conv2d/kernel/RMSProp*&
_class
loc:@disc0/conv2d/kernel*&
_output_shapes
:*
T0
?
/disc0/conv2d/kernel/RMSProp_1/Initializer/zerosConst*
dtype0*%
valueB*    *&
_output_shapes
:*&
_class
loc:@disc0/conv2d/kernel
?
disc0/conv2d/kernel/RMSProp_1
VariableV2*&
_class
loc:@disc0/conv2d/kernel*
shape:*
shared_name *
dtype0*
	container *&
_output_shapes
:
?
$disc0/conv2d/kernel/RMSProp_1/AssignAssigndisc0/conv2d/kernel/RMSProp_1/disc0/conv2d/kernel/RMSProp_1/Initializer/zeros*&
_output_shapes
:*
validate_shape(*
T0*
use_locking(*&
_class
loc:@disc0/conv2d/kernel
?
"disc0/conv2d/kernel/RMSProp_1/readIdentitydisc0/conv2d/kernel/RMSProp_1*&
_output_shapes
:*
T0*&
_class
loc:@disc0/conv2d/kernel
?
*disc0/conv2d/bias/RMSProp/Initializer/onesConst*
valueB*  ??*
_output_shapes
:*
dtype0*$
_class
loc:@disc0/conv2d/bias
?
disc0/conv2d/bias/RMSProp
VariableV2*
	container *
dtype0*$
_class
loc:@disc0/conv2d/bias*
shape:*
_output_shapes
:*
shared_name 
?
 disc0/conv2d/bias/RMSProp/AssignAssigndisc0/conv2d/bias/RMSProp*disc0/conv2d/bias/RMSProp/Initializer/ones*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*$
_class
loc:@disc0/conv2d/bias
?
disc0/conv2d/bias/RMSProp/readIdentitydisc0/conv2d/bias/RMSProp*
T0*$
_class
loc:@disc0/conv2d/bias*
_output_shapes
:
?
-disc0/conv2d/bias/RMSProp_1/Initializer/zerosConst*
_output_shapes
:*$
_class
loc:@disc0/conv2d/bias*
valueB*    *
dtype0
?
disc0/conv2d/bias/RMSProp_1
VariableV2*
dtype0*
_output_shapes
:*$
_class
loc:@disc0/conv2d/bias*
	container *
shape:*
shared_name 
?
"disc0/conv2d/bias/RMSProp_1/AssignAssigndisc0/conv2d/bias/RMSProp_1-disc0/conv2d/bias/RMSProp_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@disc0/conv2d/bias*
validate_shape(
?
 disc0/conv2d/bias/RMSProp_1/readIdentitydisc0/conv2d/bias/RMSProp_1*
T0*$
_class
loc:@disc0/conv2d/bias*
_output_shapes
:
?
>disc0/conv2d_1/kernel/RMSProp/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*(
_class
loc:@disc0/conv2d_1/kernel*%
valueB"         2   
?
4disc0/conv2d_1/kernel/RMSProp/Initializer/ones/ConstConst*
dtype0*(
_class
loc:@disc0/conv2d_1/kernel*
valueB
 *  ??*
_output_shapes
: 
?
.disc0/conv2d_1/kernel/RMSProp/Initializer/onesFill>disc0/conv2d_1/kernel/RMSProp/Initializer/ones/shape_as_tensor4disc0/conv2d_1/kernel/RMSProp/Initializer/ones/Const*(
_class
loc:@disc0/conv2d_1/kernel*

index_type0*&
_output_shapes
:2*
T0
?
disc0/conv2d_1/kernel/RMSProp
VariableV2*&
_output_shapes
:2*
shared_name *
dtype0*(
_class
loc:@disc0/conv2d_1/kernel*
	container *
shape:2
?
$disc0/conv2d_1/kernel/RMSProp/AssignAssigndisc0/conv2d_1/kernel/RMSProp.disc0/conv2d_1/kernel/RMSProp/Initializer/ones*
use_locking(*&
_output_shapes
:2*
T0*
validate_shape(*(
_class
loc:@disc0/conv2d_1/kernel
?
"disc0/conv2d_1/kernel/RMSProp/readIdentitydisc0/conv2d_1/kernel/RMSProp*&
_output_shapes
:2*
T0*(
_class
loc:@disc0/conv2d_1/kernel
?
Adisc0/conv2d_1/kernel/RMSProp_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         2   *(
_class
loc:@disc0/conv2d_1/kernel*
_output_shapes
:*
dtype0
?
7disc0/conv2d_1/kernel/RMSProp_1/Initializer/zeros/ConstConst*(
_class
loc:@disc0/conv2d_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
1disc0/conv2d_1/kernel/RMSProp_1/Initializer/zerosFillAdisc0/conv2d_1/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor7disc0/conv2d_1/kernel/RMSProp_1/Initializer/zeros/Const*&
_output_shapes
:2*

index_type0*
T0*(
_class
loc:@disc0/conv2d_1/kernel
?
disc0/conv2d_1/kernel/RMSProp_1
VariableV2*
shared_name *(
_class
loc:@disc0/conv2d_1/kernel*
shape:2*
	container *&
_output_shapes
:2*
dtype0
?
&disc0/conv2d_1/kernel/RMSProp_1/AssignAssigndisc0/conv2d_1/kernel/RMSProp_11disc0/conv2d_1/kernel/RMSProp_1/Initializer/zeros*
use_locking(*(
_class
loc:@disc0/conv2d_1/kernel*&
_output_shapes
:2*
validate_shape(*
T0
?
$disc0/conv2d_1/kernel/RMSProp_1/readIdentitydisc0/conv2d_1/kernel/RMSProp_1*&
_output_shapes
:2*
T0*(
_class
loc:@disc0/conv2d_1/kernel
?
,disc0/conv2d_1/bias/RMSProp/Initializer/onesConst*&
_class
loc:@disc0/conv2d_1/bias*
dtype0*
valueB2*  ??*
_output_shapes
:2
?
disc0/conv2d_1/bias/RMSProp
VariableV2*
shape:2*
_output_shapes
:2*
dtype0*
shared_name *&
_class
loc:@disc0/conv2d_1/bias*
	container 
?
"disc0/conv2d_1/bias/RMSProp/AssignAssigndisc0/conv2d_1/bias/RMSProp,disc0/conv2d_1/bias/RMSProp/Initializer/ones*
use_locking(*
_output_shapes
:2*
validate_shape(*
T0*&
_class
loc:@disc0/conv2d_1/bias
?
 disc0/conv2d_1/bias/RMSProp/readIdentitydisc0/conv2d_1/bias/RMSProp*&
_class
loc:@disc0/conv2d_1/bias*
_output_shapes
:2*
T0
?
/disc0/conv2d_1/bias/RMSProp_1/Initializer/zerosConst*
valueB2*    *
_output_shapes
:2*&
_class
loc:@disc0/conv2d_1/bias*
dtype0
?
disc0/conv2d_1/bias/RMSProp_1
VariableV2*
shared_name *
	container *&
_class
loc:@disc0/conv2d_1/bias*
_output_shapes
:2*
dtype0*
shape:2
?
$disc0/conv2d_1/bias/RMSProp_1/AssignAssigndisc0/conv2d_1/bias/RMSProp_1/disc0/conv2d_1/bias/RMSProp_1/Initializer/zeros*
validate_shape(*&
_class
loc:@disc0/conv2d_1/bias*
use_locking(*
_output_shapes
:2*
T0
?
"disc0/conv2d_1/bias/RMSProp_1/readIdentitydisc0/conv2d_1/bias/RMSProp_1*
_output_shapes
:2*&
_class
loc:@disc0/conv2d_1/bias*
T0
?
;disc0/dense/kernel/RMSProp/Initializer/ones/shape_as_tensorConst*
_output_shapes
:*%
_class
loc:@disc0/dense/kernel*
dtype0*
valueB"   ?  
?
1disc0/dense/kernel/RMSProp/Initializer/ones/ConstConst*
valueB
 *  ??*
_output_shapes
: *%
_class
loc:@disc0/dense/kernel*
dtype0
?
+disc0/dense/kernel/RMSProp/Initializer/onesFill;disc0/dense/kernel/RMSProp/Initializer/ones/shape_as_tensor1disc0/dense/kernel/RMSProp/Initializer/ones/Const*

index_type0* 
_output_shapes
:
??*%
_class
loc:@disc0/dense/kernel*
T0
?
disc0/dense/kernel/RMSProp
VariableV2*
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *%
_class
loc:@disc0/dense/kernel*
	container 
?
!disc0/dense/kernel/RMSProp/AssignAssigndisc0/dense/kernel/RMSProp+disc0/dense/kernel/RMSProp/Initializer/ones*
validate_shape(* 
_output_shapes
:
??*
T0*
use_locking(*%
_class
loc:@disc0/dense/kernel
?
disc0/dense/kernel/RMSProp/readIdentitydisc0/dense/kernel/RMSProp*
T0* 
_output_shapes
:
??*%
_class
loc:@disc0/dense/kernel
?
>disc0/dense/kernel/RMSProp_1/Initializer/zeros/shape_as_tensorConst*
valueB"   ?  *
dtype0*%
_class
loc:@disc0/dense/kernel*
_output_shapes
:
?
4disc0/dense/kernel/RMSProp_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*%
_class
loc:@disc0/dense/kernel*
_output_shapes
: 
?
.disc0/dense/kernel/RMSProp_1/Initializer/zerosFill>disc0/dense/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor4disc0/dense/kernel/RMSProp_1/Initializer/zeros/Const*%
_class
loc:@disc0/dense/kernel* 
_output_shapes
:
??*

index_type0*
T0
?
disc0/dense/kernel/RMSProp_1
VariableV2*
shared_name *
shape:
??*%
_class
loc:@disc0/dense/kernel*
	container *
dtype0* 
_output_shapes
:
??
?
#disc0/dense/kernel/RMSProp_1/AssignAssigndisc0/dense/kernel/RMSProp_1.disc0/dense/kernel/RMSProp_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*%
_class
loc:@disc0/dense/kernel* 
_output_shapes
:
??
?
!disc0/dense/kernel/RMSProp_1/readIdentitydisc0/dense/kernel/RMSProp_1*
T0* 
_output_shapes
:
??*%
_class
loc:@disc0/dense/kernel
?
)disc0/dense/bias/RMSProp/Initializer/onesConst*
_output_shapes	
:?*
valueB?*  ??*
dtype0*#
_class
loc:@disc0/dense/bias
?
disc0/dense/bias/RMSProp
VariableV2*#
_class
loc:@disc0/dense/bias*
	container *
shape:?*
shared_name *
dtype0*
_output_shapes	
:?
?
disc0/dense/bias/RMSProp/AssignAssigndisc0/dense/bias/RMSProp)disc0/dense/bias/RMSProp/Initializer/ones*
_output_shapes	
:?*
validate_shape(*#
_class
loc:@disc0/dense/bias*
use_locking(*
T0
?
disc0/dense/bias/RMSProp/readIdentitydisc0/dense/bias/RMSProp*
T0*#
_class
loc:@disc0/dense/bias*
_output_shapes	
:?
?
,disc0/dense/bias/RMSProp_1/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?*#
_class
loc:@disc0/dense/bias*
dtype0
?
disc0/dense/bias/RMSProp_1
VariableV2*
	container *
dtype0*
shape:?*
_output_shapes	
:?*
shared_name *#
_class
loc:@disc0/dense/bias
?
!disc0/dense/bias/RMSProp_1/AssignAssigndisc0/dense/bias/RMSProp_1,disc0/dense/bias/RMSProp_1/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(*#
_class
loc:@disc0/dense/bias
?
disc0/dense/bias/RMSProp_1/readIdentitydisc0/dense/bias/RMSProp_1*#
_class
loc:@disc0/dense/bias*
T0*
_output_shapes	
:?
?
=disc0/dense_1/kernel/RMSProp/Initializer/ones/shape_as_tensorConst*'
_class
loc:@disc0/dense_1/kernel*
_output_shapes
:*
dtype0*
valueB"?     
?
3disc0/dense_1/kernel/RMSProp/Initializer/ones/ConstConst*
_output_shapes
: *'
_class
loc:@disc0/dense_1/kernel*
valueB
 *  ??*
dtype0
?
-disc0/dense_1/kernel/RMSProp/Initializer/onesFill=disc0/dense_1/kernel/RMSProp/Initializer/ones/shape_as_tensor3disc0/dense_1/kernel/RMSProp/Initializer/ones/Const*

index_type0*
T0*'
_class
loc:@disc0/dense_1/kernel*
_output_shapes
:	?
?
disc0/dense_1/kernel/RMSProp
VariableV2*
shape:	?*
	container *
dtype0*'
_class
loc:@disc0/dense_1/kernel*
_output_shapes
:	?*
shared_name 
?
#disc0/dense_1/kernel/RMSProp/AssignAssigndisc0/dense_1/kernel/RMSProp-disc0/dense_1/kernel/RMSProp/Initializer/ones*
validate_shape(*
use_locking(*
_output_shapes
:	?*
T0*'
_class
loc:@disc0/dense_1/kernel
?
!disc0/dense_1/kernel/RMSProp/readIdentitydisc0/dense_1/kernel/RMSProp*
T0*'
_class
loc:@disc0/dense_1/kernel*
_output_shapes
:	?
?
@disc0/dense_1/kernel/RMSProp_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*'
_class
loc:@disc0/dense_1/kernel*
valueB"?     *
dtype0
?
6disc0/dense_1/kernel/RMSProp_1/Initializer/zeros/ConstConst*'
_class
loc:@disc0/dense_1/kernel*
dtype0*
_output_shapes
: *
valueB
 *    
?
0disc0/dense_1/kernel/RMSProp_1/Initializer/zerosFill@disc0/dense_1/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor6disc0/dense_1/kernel/RMSProp_1/Initializer/zeros/Const*
_output_shapes
:	?*'
_class
loc:@disc0/dense_1/kernel*
T0*

index_type0
?
disc0/dense_1/kernel/RMSProp_1
VariableV2*
_output_shapes
:	?*
dtype0*
	container *'
_class
loc:@disc0/dense_1/kernel*
shape:	?*
shared_name 
?
%disc0/dense_1/kernel/RMSProp_1/AssignAssigndisc0/dense_1/kernel/RMSProp_10disc0/dense_1/kernel/RMSProp_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*'
_class
loc:@disc0/dense_1/kernel*
_output_shapes
:	?
?
#disc0/dense_1/kernel/RMSProp_1/readIdentitydisc0/dense_1/kernel/RMSProp_1*
T0*
_output_shapes
:	?*'
_class
loc:@disc0/dense_1/kernel
?
+disc0/dense_1/bias/RMSProp/Initializer/onesConst*
_output_shapes
:*%
_class
loc:@disc0/dense_1/bias*
valueB*  ??*
dtype0
?
disc0/dense_1/bias/RMSProp
VariableV2*
	container *
shape:*%
_class
loc:@disc0/dense_1/bias*
_output_shapes
:*
dtype0*
shared_name 
?
!disc0/dense_1/bias/RMSProp/AssignAssigndisc0/dense_1/bias/RMSProp+disc0/dense_1/bias/RMSProp/Initializer/ones*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*%
_class
loc:@disc0/dense_1/bias
?
disc0/dense_1/bias/RMSProp/readIdentitydisc0/dense_1/bias/RMSProp*
_output_shapes
:*%
_class
loc:@disc0/dense_1/bias*
T0
?
.disc0/dense_1/bias/RMSProp_1/Initializer/zerosConst*
valueB*    *
dtype0*%
_class
loc:@disc0/dense_1/bias*
_output_shapes
:
?
disc0/dense_1/bias/RMSProp_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shared_name *
shape:*%
_class
loc:@disc0/dense_1/bias
?
#disc0/dense_1/bias/RMSProp_1/AssignAssigndisc0/dense_1/bias/RMSProp_1.disc0/dense_1/bias/RMSProp_1/Initializer/zeros*
_output_shapes
:*
validate_shape(*
use_locking(*%
_class
loc:@disc0/dense_1/bias*
T0
?
!disc0/dense_1/bias/RMSProp_1/readIdentitydisc0/dense_1/bias/RMSProp_1*
_output_shapes
:*%
_class
loc:@disc0/dense_1/bias*
T0
\
RMSProp_1/learning_rateConst*
dtype0*
valueB
 *??8*
_output_shapes
: 
T
RMSProp_1/decayConst*
dtype0*
_output_shapes
: *
valueB
 *fff?
W
RMSProp_1/momentumConst*
_output_shapes
: *
dtype0*
valueB
 *    
V
RMSProp_1/epsilonConst*
valueB
 *???.*
dtype0*
_output_shapes
: 
?
1RMSProp_1/update_disc0/conv2d/kernel/ApplyRMSPropApplyRMSPropdisc0/conv2d/kerneldisc0/conv2d/kernel/RMSPropdisc0/conv2d/kernel/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilongradients_2/AddN_7*
use_locking( *&
_output_shapes
:*&
_class
loc:@disc0/conv2d/kernel*
T0
?
/RMSProp_1/update_disc0/conv2d/bias/ApplyRMSPropApplyRMSPropdisc0/conv2d/biasdisc0/conv2d/bias/RMSPropdisc0/conv2d/bias/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilongradients_2/AddN_6*
use_locking( *
_output_shapes
:*$
_class
loc:@disc0/conv2d/bias*
T0
?
3RMSProp_1/update_disc0/conv2d_1/kernel/ApplyRMSPropApplyRMSPropdisc0/conv2d_1/kerneldisc0/conv2d_1/kernel/RMSPropdisc0/conv2d_1/kernel/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilongradients_2/AddN_5*
T0*&
_output_shapes
:2*
use_locking( *(
_class
loc:@disc0/conv2d_1/kernel
?
1RMSProp_1/update_disc0/conv2d_1/bias/ApplyRMSPropApplyRMSPropdisc0/conv2d_1/biasdisc0/conv2d_1/bias/RMSPropdisc0/conv2d_1/bias/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilongradients_2/AddN_4*
_output_shapes
:2*&
_class
loc:@disc0/conv2d_1/bias*
T0*
use_locking( 
?
0RMSProp_1/update_disc0/dense/kernel/ApplyRMSPropApplyRMSPropdisc0/dense/kerneldisc0/dense/kernel/RMSPropdisc0/dense/kernel/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilongradients_2/AddN_3*
T0*%
_class
loc:@disc0/dense/kernel* 
_output_shapes
:
??*
use_locking( 
?
.RMSProp_1/update_disc0/dense/bias/ApplyRMSPropApplyRMSPropdisc0/dense/biasdisc0/dense/bias/RMSPropdisc0/dense/bias/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilongradients_2/AddN_2*#
_class
loc:@disc0/dense/bias*
_output_shapes	
:?*
T0*
use_locking( 
?
2RMSProp_1/update_disc0/dense_1/kernel/ApplyRMSPropApplyRMSPropdisc0/dense_1/kerneldisc0/dense_1/kernel/RMSPropdisc0/dense_1/kernel/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilongradients_2/AddN_1*
use_locking( *
_output_shapes
:	?*'
_class
loc:@disc0/dense_1/kernel*
T0
?
0RMSProp_1/update_disc0/dense_1/bias/ApplyRMSPropApplyRMSPropdisc0/dense_1/biasdisc0/dense_1/bias/RMSPropdisc0/dense_1/bias/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilongradients_2/AddN*
T0*%
_class
loc:@disc0/dense_1/bias*
_output_shapes
:*
use_locking( 
?
	RMSProp_1NoOp0^RMSProp_1/update_disc0/conv2d/bias/ApplyRMSProp2^RMSProp_1/update_disc0/conv2d/kernel/ApplyRMSProp2^RMSProp_1/update_disc0/conv2d_1/bias/ApplyRMSProp4^RMSProp_1/update_disc0/conv2d_1/kernel/ApplyRMSProp/^RMSProp_1/update_disc0/dense/bias/ApplyRMSProp1^RMSProp_1/update_disc0/dense/kernel/ApplyRMSProp1^RMSProp_1/update_disc0/dense_1/bias/ApplyRMSProp3^RMSProp_1/update_disc0/dense_1/kernel/ApplyRMSProp
T
gradients_3/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
Z
gradients_3/grad_ys_0Const*
_output_shapes
: *
dtype0*
valueB
 *  ??
u
gradients_3/FillFillgradients_3/Shapegradients_3/grad_ys_0*

index_type0*
_output_shapes
: *
T0
B
'gradients_3/add_9_grad/tuple/group_depsNoOp^gradients_3/Fill
?
/gradients_3/add_9_grad/tuple/control_dependencyIdentitygradients_3/Fill(^gradients_3/add_9_grad/tuple/group_deps*
_output_shapes
: *#
_class
loc:@gradients_3/Fill*
T0
?
1gradients_3/add_9_grad/tuple/control_dependency_1Identitygradients_3/Fill(^gradients_3/add_9_grad/tuple/group_deps*
T0*
_output_shapes
: *#
_class
loc:@gradients_3/Fill
v
%gradients_3/Mean_6_grad/Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
?
gradients_3/Mean_6_grad/ReshapeReshape/gradients_3/add_9_grad/tuple/control_dependency%gradients_3/Mean_6_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
b
gradients_3/Mean_6_grad/ShapeShapeNeg_6*
_output_shapes
:*
out_type0*
T0
?
gradients_3/Mean_6_grad/TileTilegradients_3/Mean_6_grad/Reshapegradients_3/Mean_6_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:?????????
d
gradients_3/Mean_6_grad/Shape_1ShapeNeg_6*
T0*
out_type0*
_output_shapes
:
b
gradients_3/Mean_6_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_3/Mean_6_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
gradients_3/Mean_6_grad/ProdProdgradients_3/Mean_6_grad/Shape_1gradients_3/Mean_6_grad/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
i
gradients_3/Mean_6_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
?
gradients_3/Mean_6_grad/Prod_1Prodgradients_3/Mean_6_grad/Shape_2gradients_3/Mean_6_grad/Const_1*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
c
!gradients_3/Mean_6_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
?
gradients_3/Mean_6_grad/MaximumMaximumgradients_3/Mean_6_grad/Prod_1!gradients_3/Mean_6_grad/Maximum/y*
T0*
_output_shapes
: 
?
 gradients_3/Mean_6_grad/floordivFloorDivgradients_3/Mean_6_grad/Prodgradients_3/Mean_6_grad/Maximum*
_output_shapes
: *
T0
?
gradients_3/Mean_6_grad/CastCast gradients_3/Mean_6_grad/floordiv*

SrcT0*
_output_shapes
: *
Truncate( *

DstT0
?
gradients_3/Mean_6_grad/truedivRealDivgradients_3/Mean_6_grad/Tilegradients_3/Mean_6_grad/Cast*
T0*'
_output_shapes
:?????????
v
%gradients_3/Mean_7_grad/Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
?
gradients_3/Mean_7_grad/ReshapeReshape1gradients_3/add_9_grad/tuple/control_dependency_1%gradients_3/Mean_7_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes

:
b
gradients_3/Mean_7_grad/ShapeShapeNeg_7*
out_type0*
T0*
_output_shapes
:
?
gradients_3/Mean_7_grad/TileTilegradients_3/Mean_7_grad/Reshapegradients_3/Mean_7_grad/Shape*
T0*'
_output_shapes
:?????????*

Tmultiples0
d
gradients_3/Mean_7_grad/Shape_1ShapeNeg_7*
_output_shapes
:*
out_type0*
T0
b
gradients_3/Mean_7_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_3/Mean_7_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
gradients_3/Mean_7_grad/ProdProdgradients_3/Mean_7_grad/Shape_1gradients_3/Mean_7_grad/Const*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
i
gradients_3/Mean_7_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
?
gradients_3/Mean_7_grad/Prod_1Prodgradients_3/Mean_7_grad/Shape_2gradients_3/Mean_7_grad/Const_1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
c
!gradients_3/Mean_7_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
?
gradients_3/Mean_7_grad/MaximumMaximumgradients_3/Mean_7_grad/Prod_1!gradients_3/Mean_7_grad/Maximum/y*
_output_shapes
: *
T0
?
 gradients_3/Mean_7_grad/floordivFloorDivgradients_3/Mean_7_grad/Prodgradients_3/Mean_7_grad/Maximum*
T0*
_output_shapes
: 
?
gradients_3/Mean_7_grad/CastCast gradients_3/Mean_7_grad/floordiv*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0
?
gradients_3/Mean_7_grad/truedivRealDivgradients_3/Mean_7_grad/Tilegradients_3/Mean_7_grad/Cast*'
_output_shapes
:?????????*
T0
t
gradients_3/Neg_6_grad/NegNeggradients_3/Mean_6_grad/truediv*
T0*'
_output_shapes
:?????????
t
gradients_3/Neg_7_grad/NegNeggradients_3/Mean_7_grad/truediv*'
_output_shapes
:?????????*
T0
?
!gradients_3/Log_4_grad/Reciprocal
Reciprocaladd_7^gradients_3/Neg_6_grad/Neg*'
_output_shapes
:?????????*
T0
?
gradients_3/Log_4_grad/mulMulgradients_3/Neg_6_grad/Neg!gradients_3/Log_4_grad/Reciprocal*
T0*'
_output_shapes
:?????????
?
!gradients_3/Log_5_grad/Reciprocal
Reciprocaladd_8^gradients_3/Neg_7_grad/Neg*'
_output_shapes
:?????????*
T0
?
gradients_3/Log_5_grad/mulMulgradients_3/Neg_7_grad/Neg!gradients_3/Log_5_grad/Reciprocal*
T0*'
_output_shapes
:?????????
a
gradients_3/add_7_grad/ShapeShapesub_2*
out_type0*
_output_shapes
:*
T0
a
gradients_3/add_7_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
?
,gradients_3/add_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/add_7_grad/Shapegradients_3/add_7_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_3/add_7_grad/SumSumgradients_3/Log_4_grad/mul,gradients_3/add_7_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
?
gradients_3/add_7_grad/ReshapeReshapegradients_3/add_7_grad/Sumgradients_3/add_7_grad/Shape*
T0*'
_output_shapes
:?????????*
Tshape0
?
gradients_3/add_7_grad/Sum_1Sumgradients_3/Log_4_grad/mul.gradients_3/add_7_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
?
 gradients_3/add_7_grad/Reshape_1Reshapegradients_3/add_7_grad/Sum_1gradients_3/add_7_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_3/add_7_grad/tuple/group_depsNoOp^gradients_3/add_7_grad/Reshape!^gradients_3/add_7_grad/Reshape_1
?
/gradients_3/add_7_grad/tuple/control_dependencyIdentitygradients_3/add_7_grad/Reshape(^gradients_3/add_7_grad/tuple/group_deps*1
_class'
%#loc:@gradients_3/add_7_grad/Reshape*
T0*'
_output_shapes
:?????????
?
1gradients_3/add_7_grad/tuple/control_dependency_1Identity gradients_3/add_7_grad/Reshape_1(^gradients_3/add_7_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_3/add_7_grad/Reshape_1*
_output_shapes
: 
e
gradients_3/add_8_grad/ShapeShape	Sigmoid_1*
T0*
_output_shapes
:*
out_type0
a
gradients_3/add_8_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
,gradients_3/add_8_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/add_8_grad/Shapegradients_3/add_8_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_3/add_8_grad/SumSumgradients_3/Log_5_grad/mul,gradients_3/add_8_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
gradients_3/add_8_grad/ReshapeReshapegradients_3/add_8_grad/Sumgradients_3/add_8_grad/Shape*
Tshape0*'
_output_shapes
:?????????*
T0
?
gradients_3/add_8_grad/Sum_1Sumgradients_3/Log_5_grad/mul.gradients_3/add_8_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
?
 gradients_3/add_8_grad/Reshape_1Reshapegradients_3/add_8_grad/Sum_1gradients_3/add_8_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
s
'gradients_3/add_8_grad/tuple/group_depsNoOp^gradients_3/add_8_grad/Reshape!^gradients_3/add_8_grad/Reshape_1
?
/gradients_3/add_8_grad/tuple/control_dependencyIdentitygradients_3/add_8_grad/Reshape(^gradients_3/add_8_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*1
_class'
%#loc:@gradients_3/add_8_grad/Reshape
?
1gradients_3/add_8_grad/tuple/control_dependency_1Identity gradients_3/add_8_grad/Reshape_1(^gradients_3/add_8_grad/tuple/group_deps*
T0*
_output_shapes
: *3
_class)
'%loc:@gradients_3/add_8_grad/Reshape_1
_
gradients_3/sub_2_grad/ShapeConst*
valueB *
_output_shapes
: *
dtype0
g
gradients_3/sub_2_grad/Shape_1Shape	Sigmoid_3*
T0*
out_type0*
_output_shapes
:
?
,gradients_3/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/sub_2_grad/Shapegradients_3/sub_2_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_3/sub_2_grad/SumSum/gradients_3/add_7_grad/tuple/control_dependency,gradients_3/sub_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
?
gradients_3/sub_2_grad/ReshapeReshapegradients_3/sub_2_grad/Sumgradients_3/sub_2_grad/Shape*
_output_shapes
: *
Tshape0*
T0
?
gradients_3/sub_2_grad/Sum_1Sum/gradients_3/add_7_grad/tuple/control_dependency.gradients_3/sub_2_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
b
gradients_3/sub_2_grad/NegNeggradients_3/sub_2_grad/Sum_1*
T0*
_output_shapes
:
?
 gradients_3/sub_2_grad/Reshape_1Reshapegradients_3/sub_2_grad/Neggradients_3/sub_2_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:?????????
s
'gradients_3/sub_2_grad/tuple/group_depsNoOp^gradients_3/sub_2_grad/Reshape!^gradients_3/sub_2_grad/Reshape_1
?
/gradients_3/sub_2_grad/tuple/control_dependencyIdentitygradients_3/sub_2_grad/Reshape(^gradients_3/sub_2_grad/tuple/group_deps*
T0*
_output_shapes
: *1
_class'
%#loc:@gradients_3/sub_2_grad/Reshape
?
1gradients_3/sub_2_grad/tuple/control_dependency_1Identity gradients_3/sub_2_grad/Reshape_1(^gradients_3/sub_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_3/sub_2_grad/Reshape_1*'
_output_shapes
:?????????
?
&gradients_3/Sigmoid_1_grad/SigmoidGradSigmoidGrad	Sigmoid_1/gradients_3/add_8_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????
?
&gradients_3/Sigmoid_3_grad/SigmoidGradSigmoidGrad	Sigmoid_31gradients_3/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
2gradients_3/diz_1/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad&gradients_3/Sigmoid_1_grad/SigmoidGrad*
data_formatNHWC*
T0*
_output_shapes
:
?
7gradients_3/diz_1/dense_3/BiasAdd_grad/tuple/group_depsNoOp'^gradients_3/Sigmoid_1_grad/SigmoidGrad3^gradients_3/diz_1/dense_3/BiasAdd_grad/BiasAddGrad
?
?gradients_3/diz_1/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity&gradients_3/Sigmoid_1_grad/SigmoidGrad8^gradients_3/diz_1/dense_3/BiasAdd_grad/tuple/group_deps*9
_class/
-+loc:@gradients_3/Sigmoid_1_grad/SigmoidGrad*'
_output_shapes
:?????????*
T0
?
Agradients_3/diz_1/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_3/diz_1/dense_3/BiasAdd_grad/BiasAddGrad8^gradients_3/diz_1/dense_3/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*E
_class;
97loc:@gradients_3/diz_1/dense_3/BiasAdd_grad/BiasAddGrad*
T0
?
4gradients_3/disc0_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad&gradients_3/Sigmoid_3_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
?
9gradients_3/disc0_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp'^gradients_3/Sigmoid_3_grad/SigmoidGrad5^gradients_3/disc0_1/dense_1/BiasAdd_grad/BiasAddGrad
?
Agradients_3/disc0_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity&gradients_3/Sigmoid_3_grad/SigmoidGrad:^gradients_3/disc0_1/dense_1/BiasAdd_grad/tuple/group_deps*9
_class/
-+loc:@gradients_3/Sigmoid_3_grad/SigmoidGrad*
T0*'
_output_shapes
:?????????
?
Cgradients_3/disc0_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_3/disc0_1/dense_1/BiasAdd_grad/BiasAddGrad:^gradients_3/disc0_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*G
_class=
;9loc:@gradients_3/disc0_1/dense_1/BiasAdd_grad/BiasAddGrad
?
,gradients_3/diz_1/dense_3/MatMul_grad/MatMulMatMul?gradients_3/diz_1/dense_3/BiasAdd_grad/tuple/control_dependencydiz/dense_3/kernel/read*'
_output_shapes
:????????? *
transpose_b(*
T0*
transpose_a( 
?
.gradients_3/diz_1/dense_3/MatMul_grad/MatMul_1MatMuldiz_1/dense_2/LeakyRelu?gradients_3/diz_1/dense_3/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
T0*
transpose_b( *
_output_shapes

: 
?
6gradients_3/diz_1/dense_3/MatMul_grad/tuple/group_depsNoOp-^gradients_3/diz_1/dense_3/MatMul_grad/MatMul/^gradients_3/diz_1/dense_3/MatMul_grad/MatMul_1
?
>gradients_3/diz_1/dense_3/MatMul_grad/tuple/control_dependencyIdentity,gradients_3/diz_1/dense_3/MatMul_grad/MatMul7^gradients_3/diz_1/dense_3/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_3/diz_1/dense_3/MatMul_grad/MatMul*'
_output_shapes
:????????? 
?
@gradients_3/diz_1/dense_3/MatMul_grad/tuple/control_dependency_1Identity.gradients_3/diz_1/dense_3/MatMul_grad/MatMul_17^gradients_3/diz_1/dense_3/MatMul_grad/tuple/group_deps*A
_class7
53loc:@gradients_3/diz_1/dense_3/MatMul_grad/MatMul_1*
_output_shapes

: *
T0
?
.gradients_3/disc0_1/dense_1/MatMul_grad/MatMulMatMulAgradients_3/disc0_1/dense_1/BiasAdd_grad/tuple/control_dependencydisc0/dense_1/kernel/read*
transpose_b(*
T0*(
_output_shapes
:??????????*
transpose_a( 
?
0gradients_3/disc0_1/dense_1/MatMul_grad/MatMul_1MatMuldisc0_1/dense/ReluAgradients_3/disc0_1/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes
:	?*
T0*
transpose_a(
?
8gradients_3/disc0_1/dense_1/MatMul_grad/tuple/group_depsNoOp/^gradients_3/disc0_1/dense_1/MatMul_grad/MatMul1^gradients_3/disc0_1/dense_1/MatMul_grad/MatMul_1
?
@gradients_3/disc0_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity.gradients_3/disc0_1/dense_1/MatMul_grad/MatMul9^gradients_3/disc0_1/dense_1/MatMul_grad/tuple/group_deps*A
_class7
53loc:@gradients_3/disc0_1/dense_1/MatMul_grad/MatMul*
T0*(
_output_shapes
:??????????
?
Bgradients_3/disc0_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity0gradients_3/disc0_1/dense_1/MatMul_grad/MatMul_19^gradients_3/disc0_1/dense_1/MatMul_grad/tuple/group_deps*C
_class9
75loc:@gradients_3/disc0_1/dense_1/MatMul_grad/MatMul_1*
_output_shapes
:	?*
T0
?
6gradients_3/diz_1/dense_2/LeakyRelu_grad/LeakyReluGradLeakyReluGrad>gradients_3/diz_1/dense_3/MatMul_grad/tuple/control_dependencydiz_1/dense_2/BiasAdd*
T0*
alpha%??L>*'
_output_shapes
:????????? 
?
,gradients_3/disc0_1/dense/Relu_grad/ReluGradReluGrad@gradients_3/disc0_1/dense_1/MatMul_grad/tuple/control_dependencydisc0_1/dense/Relu*
T0*(
_output_shapes
:??????????
?
2gradients_3/diz_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients_3/diz_1/dense_2/LeakyRelu_grad/LeakyReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
?
7gradients_3/diz_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp3^gradients_3/diz_1/dense_2/BiasAdd_grad/BiasAddGrad7^gradients_3/diz_1/dense_2/LeakyRelu_grad/LeakyReluGrad
?
?gradients_3/diz_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity6gradients_3/diz_1/dense_2/LeakyRelu_grad/LeakyReluGrad8^gradients_3/diz_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:????????? *I
_class?
=;loc:@gradients_3/diz_1/dense_2/LeakyRelu_grad/LeakyReluGrad
?
Agradients_3/diz_1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_3/diz_1/dense_2/BiasAdd_grad/BiasAddGrad8^gradients_3/diz_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
: *E
_class;
97loc:@gradients_3/diz_1/dense_2/BiasAdd_grad/BiasAddGrad
?
2gradients_3/disc0_1/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_3/disc0_1/dense/Relu_grad/ReluGrad*
_output_shapes	
:?*
data_formatNHWC*
T0
?
7gradients_3/disc0_1/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients_3/disc0_1/dense/BiasAdd_grad/BiasAddGrad-^gradients_3/disc0_1/dense/Relu_grad/ReluGrad
?
?gradients_3/disc0_1/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients_3/disc0_1/dense/Relu_grad/ReluGrad8^gradients_3/disc0_1/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:??????????*
T0*?
_class5
31loc:@gradients_3/disc0_1/dense/Relu_grad/ReluGrad
?
Agradients_3/disc0_1/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_3/disc0_1/dense/BiasAdd_grad/BiasAddGrad8^gradients_3/disc0_1/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:?*
T0*E
_class;
97loc:@gradients_3/disc0_1/dense/BiasAdd_grad/BiasAddGrad
?
,gradients_3/diz_1/dense_2/MatMul_grad/MatMulMatMul?gradients_3/diz_1/dense_2/BiasAdd_grad/tuple/control_dependencydiz/dense_2/kernel/read*'
_output_shapes
:????????? *
transpose_a( *
transpose_b(*
T0
?
.gradients_3/diz_1/dense_2/MatMul_grad/MatMul_1MatMuldiz_1/dense_1/LeakyRelu?gradients_3/diz_1/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes

:  
?
6gradients_3/diz_1/dense_2/MatMul_grad/tuple/group_depsNoOp-^gradients_3/diz_1/dense_2/MatMul_grad/MatMul/^gradients_3/diz_1/dense_2/MatMul_grad/MatMul_1
?
>gradients_3/diz_1/dense_2/MatMul_grad/tuple/control_dependencyIdentity,gradients_3/diz_1/dense_2/MatMul_grad/MatMul7^gradients_3/diz_1/dense_2/MatMul_grad/tuple/group_deps*'
_output_shapes
:????????? *?
_class5
31loc:@gradients_3/diz_1/dense_2/MatMul_grad/MatMul*
T0
?
@gradients_3/diz_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity.gradients_3/diz_1/dense_2/MatMul_grad/MatMul_17^gradients_3/diz_1/dense_2/MatMul_grad/tuple/group_deps*A
_class7
53loc:@gradients_3/diz_1/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:  *
T0
?
,gradients_3/disc0_1/dense/MatMul_grad/MatMulMatMul?gradients_3/disc0_1/dense/BiasAdd_grad/tuple/control_dependencydisc0/dense/kernel/read*
transpose_b(*(
_output_shapes
:??????????*
T0*
transpose_a( 
?
.gradients_3/disc0_1/dense/MatMul_grad/MatMul_1MatMuldisc0_1/flatten/Reshape?gradients_3/disc0_1/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( * 
_output_shapes
:
??*
transpose_a(
?
6gradients_3/disc0_1/dense/MatMul_grad/tuple/group_depsNoOp-^gradients_3/disc0_1/dense/MatMul_grad/MatMul/^gradients_3/disc0_1/dense/MatMul_grad/MatMul_1
?
>gradients_3/disc0_1/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients_3/disc0_1/dense/MatMul_grad/MatMul7^gradients_3/disc0_1/dense/MatMul_grad/tuple/group_deps*?
_class5
31loc:@gradients_3/disc0_1/dense/MatMul_grad/MatMul*
T0*(
_output_shapes
:??????????
?
@gradients_3/disc0_1/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients_3/disc0_1/dense/MatMul_grad/MatMul_17^gradients_3/disc0_1/dense/MatMul_grad/tuple/group_deps* 
_output_shapes
:
??*A
_class7
53loc:@gradients_3/disc0_1/dense/MatMul_grad/MatMul_1*
T0
?
6gradients_3/diz_1/dense_1/LeakyRelu_grad/LeakyReluGradLeakyReluGrad>gradients_3/diz_1/dense_2/MatMul_grad/tuple/control_dependencydiz_1/dense_1/BiasAdd*
T0*
alpha%??L>*'
_output_shapes
:????????? 

.gradients_3/disc0_1/flatten/Reshape_grad/ShapeShapedisc0_1/MaxPool_1*
T0*
out_type0*
_output_shapes
:
?
0gradients_3/disc0_1/flatten/Reshape_grad/ReshapeReshape>gradients_3/disc0_1/dense/MatMul_grad/tuple/control_dependency.gradients_3/disc0_1/flatten/Reshape_grad/Shape*
T0*/
_output_shapes
:?????????2*
Tshape0
?
2gradients_3/diz_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients_3/diz_1/dense_1/LeakyRelu_grad/LeakyReluGrad*
T0*
_output_shapes
: *
data_formatNHWC
?
7gradients_3/diz_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp3^gradients_3/diz_1/dense_1/BiasAdd_grad/BiasAddGrad7^gradients_3/diz_1/dense_1/LeakyRelu_grad/LeakyReluGrad
?
?gradients_3/diz_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients_3/diz_1/dense_1/LeakyRelu_grad/LeakyReluGrad8^gradients_3/diz_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_3/diz_1/dense_1/LeakyRelu_grad/LeakyReluGrad*'
_output_shapes
:????????? 
?
Agradients_3/diz_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_3/diz_1/dense_1/BiasAdd_grad/BiasAddGrad8^gradients_3/diz_1/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
: *
T0*E
_class;
97loc:@gradients_3/diz_1/dense_1/BiasAdd_grad/BiasAddGrad
?
.gradients_3/disc0_1/MaxPool_1_grad/MaxPoolGradMaxPoolGraddisc0_1/conv2d_1/Reludisc0_1/MaxPool_10gradients_3/disc0_1/flatten/Reshape_grad/Reshape*
ksize
*/
_output_shapes
:?????????2*
data_formatNHWC*
T0*
paddingSAME*
strides

?
,gradients_3/diz_1/dense_1/MatMul_grad/MatMulMatMul?gradients_3/diz_1/dense_1/BiasAdd_grad/tuple/control_dependencydiz/dense_1/kernel/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:????????? 
?
.gradients_3/diz_1/dense_1/MatMul_grad/MatMul_1MatMuldiz_1/dense/LeakyRelu?gradients_3/diz_1/dense_1/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:  *
transpose_b( 
?
6gradients_3/diz_1/dense_1/MatMul_grad/tuple/group_depsNoOp-^gradients_3/diz_1/dense_1/MatMul_grad/MatMul/^gradients_3/diz_1/dense_1/MatMul_grad/MatMul_1
?
>gradients_3/diz_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity,gradients_3/diz_1/dense_1/MatMul_grad/MatMul7^gradients_3/diz_1/dense_1/MatMul_grad/tuple/group_deps*?
_class5
31loc:@gradients_3/diz_1/dense_1/MatMul_grad/MatMul*
T0*'
_output_shapes
:????????? 
?
@gradients_3/diz_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity.gradients_3/diz_1/dense_1/MatMul_grad/MatMul_17^gradients_3/diz_1/dense_1/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_3/diz_1/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:  
?
/gradients_3/disc0_1/conv2d_1/Relu_grad/ReluGradReluGrad.gradients_3/disc0_1/MaxPool_1_grad/MaxPoolGraddisc0_1/conv2d_1/Relu*
T0*/
_output_shapes
:?????????2
?
4gradients_3/diz_1/dense/LeakyRelu_grad/LeakyReluGradLeakyReluGrad>gradients_3/diz_1/dense_1/MatMul_grad/tuple/control_dependencydiz_1/dense/BiasAdd*'
_output_shapes
:????????? *
T0*
alpha%??L>
?
5gradients_3/disc0_1/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad/gradients_3/disc0_1/conv2d_1/Relu_grad/ReluGrad*
_output_shapes
:2*
T0*
data_formatNHWC
?
:gradients_3/disc0_1/conv2d_1/BiasAdd_grad/tuple/group_depsNoOp6^gradients_3/disc0_1/conv2d_1/BiasAdd_grad/BiasAddGrad0^gradients_3/disc0_1/conv2d_1/Relu_grad/ReluGrad
?
Bgradients_3/disc0_1/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity/gradients_3/disc0_1/conv2d_1/Relu_grad/ReluGrad;^gradients_3/disc0_1/conv2d_1/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:?????????2*B
_class8
64loc:@gradients_3/disc0_1/conv2d_1/Relu_grad/ReluGrad*
T0
?
Dgradients_3/disc0_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity5gradients_3/disc0_1/conv2d_1/BiasAdd_grad/BiasAddGrad;^gradients_3/disc0_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_3/disc0_1/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:2
?
0gradients_3/diz_1/dense/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients_3/diz_1/dense/LeakyRelu_grad/LeakyReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
?
5gradients_3/diz_1/dense/BiasAdd_grad/tuple/group_depsNoOp1^gradients_3/diz_1/dense/BiasAdd_grad/BiasAddGrad5^gradients_3/diz_1/dense/LeakyRelu_grad/LeakyReluGrad
?
=gradients_3/diz_1/dense/BiasAdd_grad/tuple/control_dependencyIdentity4gradients_3/diz_1/dense/LeakyRelu_grad/LeakyReluGrad6^gradients_3/diz_1/dense/BiasAdd_grad/tuple/group_deps*G
_class=
;9loc:@gradients_3/diz_1/dense/LeakyRelu_grad/LeakyReluGrad*
T0*'
_output_shapes
:????????? 
?
?gradients_3/diz_1/dense/BiasAdd_grad/tuple/control_dependency_1Identity0gradients_3/diz_1/dense/BiasAdd_grad/BiasAddGrad6^gradients_3/diz_1/dense/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@gradients_3/diz_1/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
?
/gradients_3/disc0_1/conv2d_1/Conv2D_grad/ShapeNShapeNdisc0_1/MaxPooldisc0/conv2d_1/kernel/read*
N*
out_type0* 
_output_shapes
::*
T0
?
<gradients_3/disc0_1/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput/gradients_3/disc0_1/conv2d_1/Conv2D_grad/ShapeNdisc0/conv2d_1/kernel/readBgradients_3/disc0_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*
data_formatNHWC*/
_output_shapes
:?????????*
	dilations
*
strides
*
T0
?
=gradients_3/disc0_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdisc0_1/MaxPool1gradients_3/disc0_1/conv2d_1/Conv2D_grad/ShapeN:1Bgradients_3/disc0_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
:2*
use_cudnn_on_gpu(*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 
?
9gradients_3/disc0_1/conv2d_1/Conv2D_grad/tuple/group_depsNoOp>^gradients_3/disc0_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilter=^gradients_3/disc0_1/conv2d_1/Conv2D_grad/Conv2DBackpropInput
?
Agradients_3/disc0_1/conv2d_1/Conv2D_grad/tuple/control_dependencyIdentity<gradients_3/disc0_1/conv2d_1/Conv2D_grad/Conv2DBackpropInput:^gradients_3/disc0_1/conv2d_1/Conv2D_grad/tuple/group_deps*/
_output_shapes
:?????????*O
_classE
CAloc:@gradients_3/disc0_1/conv2d_1/Conv2D_grad/Conv2DBackpropInput*
T0
?
Cgradients_3/disc0_1/conv2d_1/Conv2D_grad/tuple/control_dependency_1Identity=gradients_3/disc0_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilter:^gradients_3/disc0_1/conv2d_1/Conv2D_grad/tuple/group_deps*P
_classF
DBloc:@gradients_3/disc0_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:2*
T0
?
*gradients_3/diz_1/dense/MatMul_grad/MatMulMatMul=gradients_3/diz_1/dense/BiasAdd_grad/tuple/control_dependencydiz/dense/kernel/read*
transpose_a( *
transpose_b(*'
_output_shapes
:?????????d*
T0
?
,gradients_3/diz_1/dense/MatMul_grad/MatMul_1MatMul@MultivariateNormalDiag/sample/affine_linear_operator/forward/add=gradients_3/diz_1/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
_output_shapes

:d *
T0
?
4gradients_3/diz_1/dense/MatMul_grad/tuple/group_depsNoOp+^gradients_3/diz_1/dense/MatMul_grad/MatMul-^gradients_3/diz_1/dense/MatMul_grad/MatMul_1
?
<gradients_3/diz_1/dense/MatMul_grad/tuple/control_dependencyIdentity*gradients_3/diz_1/dense/MatMul_grad/MatMul5^gradients_3/diz_1/dense/MatMul_grad/tuple/group_deps*=
_class3
1/loc:@gradients_3/diz_1/dense/MatMul_grad/MatMul*
T0*'
_output_shapes
:?????????d
?
>gradients_3/diz_1/dense/MatMul_grad/tuple/control_dependency_1Identity,gradients_3/diz_1/dense/MatMul_grad/MatMul_15^gradients_3/diz_1/dense/MatMul_grad/tuple/group_deps*?
_class5
31loc:@gradients_3/diz_1/dense/MatMul_grad/MatMul_1*
T0*
_output_shapes

:d 
?
,gradients_3/disc0_1/MaxPool_grad/MaxPoolGradMaxPoolGraddisc0_1/conv2d/Reludisc0_1/MaxPoolAgradients_3/disc0_1/conv2d_1/Conv2D_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
ksize
*/
_output_shapes
:?????????*
paddingSAME*
T0
?
-gradients_3/disc0_1/conv2d/Relu_grad/ReluGradReluGrad,gradients_3/disc0_1/MaxPool_grad/MaxPoolGraddisc0_1/conv2d/Relu*/
_output_shapes
:?????????*
T0
?
3gradients_3/disc0_1/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients_3/disc0_1/conv2d/Relu_grad/ReluGrad*
_output_shapes
:*
data_formatNHWC*
T0
?
8gradients_3/disc0_1/conv2d/BiasAdd_grad/tuple/group_depsNoOp4^gradients_3/disc0_1/conv2d/BiasAdd_grad/BiasAddGrad.^gradients_3/disc0_1/conv2d/Relu_grad/ReluGrad
?
@gradients_3/disc0_1/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity-gradients_3/disc0_1/conv2d/Relu_grad/ReluGrad9^gradients_3/disc0_1/conv2d/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_3/disc0_1/conv2d/Relu_grad/ReluGrad*/
_output_shapes
:?????????
?
Bgradients_3/disc0_1/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity3gradients_3/disc0_1/conv2d/BiasAdd_grad/BiasAddGrad9^gradients_3/disc0_1/conv2d/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*F
_class<
:8loc:@gradients_3/disc0_1/conv2d/BiasAdd_grad/BiasAddGrad*
T0
?
-gradients_3/disc0_1/conv2d/Conv2D_grad/ShapeNShapeNdisc0_1/Reshapedisc0/conv2d/kernel/read*
T0*
N*
out_type0* 
_output_shapes
::
?
:gradients_3/disc0_1/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput-gradients_3/disc0_1/conv2d/Conv2D_grad/ShapeNdisc0/conv2d/kernel/read@gradients_3/disc0_1/conv2d/BiasAdd_grad/tuple/control_dependency*
explicit_paddings
 *
data_formatNHWC*
paddingVALID*
T0*
strides
*
use_cudnn_on_gpu(*/
_output_shapes
:?????????*
	dilations

?
;gradients_3/disc0_1/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdisc0_1/Reshape/gradients_3/disc0_1/conv2d/Conv2D_grad/ShapeN:1@gradients_3/disc0_1/conv2d/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
explicit_paddings
 *
paddingVALID*
T0*&
_output_shapes
:*
	dilations

?
7gradients_3/disc0_1/conv2d/Conv2D_grad/tuple/group_depsNoOp<^gradients_3/disc0_1/conv2d/Conv2D_grad/Conv2DBackpropFilter;^gradients_3/disc0_1/conv2d/Conv2D_grad/Conv2DBackpropInput
?
?gradients_3/disc0_1/conv2d/Conv2D_grad/tuple/control_dependencyIdentity:gradients_3/disc0_1/conv2d/Conv2D_grad/Conv2DBackpropInput8^gradients_3/disc0_1/conv2d/Conv2D_grad/tuple/group_deps*M
_classC
A?loc:@gradients_3/disc0_1/conv2d/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????*
T0
?
Agradients_3/disc0_1/conv2d/Conv2D_grad/tuple/control_dependency_1Identity;gradients_3/disc0_1/conv2d/Conv2D_grad/Conv2DBackpropFilter8^gradients_3/disc0_1/conv2d/Conv2D_grad/tuple/group_deps*&
_output_shapes
:*
T0*N
_classD
B@loc:@gradients_3/disc0_1/conv2d/Conv2D_grad/Conv2DBackpropFilter
?
&gradients_3/disc0_1/Reshape_grad/ShapeShapedecoder_1/generated_images*
_output_shapes
:*
out_type0*
T0
?
(gradients_3/disc0_1/Reshape_grad/ReshapeReshape?gradients_3/disc0_1/conv2d/Conv2D_grad/tuple/control_dependency&gradients_3/disc0_1/Reshape_grad/Shape*
Tshape0*
T0*/
_output_shapes
:?????????
?
0gradients_3/decoder_1/layer_2/Tanh_grad/TanhGradTanhGraddecoder_1/layer_2/Tanh(gradients_3/disc0_1/Reshape_grad/Reshape*/
_output_shapes
:?????????*
T0
?
6gradients_3/decoder_1/layer_2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients_3/decoder_1/layer_2/Tanh_grad/TanhGrad*
_output_shapes
:*
data_formatNHWC*
T0
?
;gradients_3/decoder_1/layer_2/BiasAdd_grad/tuple/group_depsNoOp7^gradients_3/decoder_1/layer_2/BiasAdd_grad/BiasAddGrad1^gradients_3/decoder_1/layer_2/Tanh_grad/TanhGrad
?
Cgradients_3/decoder_1/layer_2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients_3/decoder_1/layer_2/Tanh_grad/TanhGrad<^gradients_3/decoder_1/layer_2/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:?????????*C
_class9
75loc:@gradients_3/decoder_1/layer_2/Tanh_grad/TanhGrad*
T0
?
Egradients_3/decoder_1/layer_2/BiasAdd_grad/tuple/control_dependency_1Identity6gradients_3/decoder_1/layer_2/BiasAdd_grad/BiasAddGrad<^gradients_3/decoder_1/layer_2/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*I
_class?
=;loc:@gradients_3/decoder_1/layer_2/BiasAdd_grad/BiasAddGrad
?
9gradients_3/decoder_1/layer_2/conv2d_transpose_grad/ShapeConst*
dtype0*%
valueB"         ?   *
_output_shapes
:
?
Hgradients_3/decoder_1/layer_2/conv2d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilterCgradients_3/decoder_1/layer_2/BiasAdd_grad/tuple/control_dependency9gradients_3/decoder_1/layer_2/conv2d_transpose_grad/Shapedecoder_1/LeakyRelu_1*
	dilations
*'
_output_shapes
:?*
T0*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*
strides
*
data_formatNHWC
?
:gradients_3/decoder_1/layer_2/conv2d_transpose_grad/Conv2DConv2DCgradients_3/decoder_1/layer_2/BiasAdd_grad/tuple/control_dependencydecoder/layer_2/kernel/read*
data_formatNHWC*
strides
*0
_output_shapes
:??????????*
	dilations
*
T0*
use_cudnn_on_gpu(*
paddingSAME*
explicit_paddings
 
?
Dgradients_3/decoder_1/layer_2/conv2d_transpose_grad/tuple/group_depsNoOp;^gradients_3/decoder_1/layer_2/conv2d_transpose_grad/Conv2DI^gradients_3/decoder_1/layer_2/conv2d_transpose_grad/Conv2DBackpropFilter
?
Lgradients_3/decoder_1/layer_2/conv2d_transpose_grad/tuple/control_dependencyIdentityHgradients_3/decoder_1/layer_2/conv2d_transpose_grad/Conv2DBackpropFilterE^gradients_3/decoder_1/layer_2/conv2d_transpose_grad/tuple/group_deps*'
_output_shapes
:?*[
_classQ
OMloc:@gradients_3/decoder_1/layer_2/conv2d_transpose_grad/Conv2DBackpropFilter*
T0
?
Ngradients_3/decoder_1/layer_2/conv2d_transpose_grad/tuple/control_dependency_1Identity:gradients_3/decoder_1/layer_2/conv2d_transpose_grad/Conv2DE^gradients_3/decoder_1/layer_2/conv2d_transpose_grad/tuple/group_deps*
T0*0
_output_shapes
:??????????*M
_classC
A?loc:@gradients_3/decoder_1/layer_2/conv2d_transpose_grad/Conv2D
?
4gradients_3/decoder_1/LeakyRelu_1_grad/LeakyReluGradLeakyReluGradNgradients_3/decoder_1/layer_2/conv2d_transpose_grad/tuple/control_dependency_1.decoder_1/batch_normalization_1/FusedBatchNorm*0
_output_shapes
:??????????*
T0*
alpha%??L>
{
gradients_3/zeros_like	ZerosLike0decoder_1/batch_normalization_1/FusedBatchNorm:1*
T0*
_output_shapes	
:?
}
gradients_3/zeros_like_1	ZerosLike0decoder_1/batch_normalization_1/FusedBatchNorm:2*
T0*
_output_shapes	
:?
}
gradients_3/zeros_like_2	ZerosLike0decoder_1/batch_normalization_1/FusedBatchNorm:3*
T0*
_output_shapes	
:?
}
gradients_3/zeros_like_3	ZerosLike0decoder_1/batch_normalization_1/FusedBatchNorm:4*
_output_shapes	
:?*
T0
?
Rgradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGrad4gradients_3/decoder_1/LeakyRelu_1_grad/LeakyReluGraddecoder_1/layer_1/BiasAdd(decoder/batch_normalization_1/gamma/read0decoder_1/batch_normalization_1/FusedBatchNorm:30decoder_1/batch_normalization_1/FusedBatchNorm:4*
T0*
is_training(*
epsilon%o?:*
data_formatNHWC*F
_output_shapes4
2:??????????:?:?: : 
?
Pgradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/tuple/group_depsNoOpS^gradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad
?
Xgradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependencyIdentityRgradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGradQ^gradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*0
_output_shapes
:??????????*e
_class[
YWloc:@gradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*
T0
?
Zgradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_1IdentityTgradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad:1Q^gradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
T0*
_output_shapes	
:?*e
_class[
YWloc:@gradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad
?
Zgradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_2IdentityTgradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad:2Q^gradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes	
:?*e
_class[
YWloc:@gradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*
T0
?
Zgradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_3IdentityTgradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad:3Q^gradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*e
_class[
YWloc:@gradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
?
Zgradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_4IdentityTgradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad:4Q^gradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*e
_class[
YWloc:@gradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
?
6gradients_3/decoder_1/layer_1/BiasAdd_grad/BiasAddGradBiasAddGradXgradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency*
T0*
_output_shapes	
:?*
data_formatNHWC
?
;gradients_3/decoder_1/layer_1/BiasAdd_grad/tuple/group_depsNoOpY^gradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency7^gradients_3/decoder_1/layer_1/BiasAdd_grad/BiasAddGrad
?
Cgradients_3/decoder_1/layer_1/BiasAdd_grad/tuple/control_dependencyIdentityXgradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency<^gradients_3/decoder_1/layer_1/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:??????????*e
_class[
YWloc:@gradients_3/decoder_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*
T0
?
Egradients_3/decoder_1/layer_1/BiasAdd_grad/tuple/control_dependency_1Identity6gradients_3/decoder_1/layer_1/BiasAdd_grad/BiasAddGrad<^gradients_3/decoder_1/layer_1/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_3/decoder_1/layer_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
9gradients_3/decoder_1/layer_1/conv2d_transpose_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?      
?
Hgradients_3/decoder_1/layer_1/conv2d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilterCgradients_3/decoder_1/layer_1/BiasAdd_grad/tuple/control_dependency9gradients_3/decoder_1/layer_1/conv2d_transpose_grad/Shapedecoder_1/LeakyRelu*
paddingSAME*
data_formatNHWC*
T0*
strides
*
	dilations
*
explicit_paddings
 *
use_cudnn_on_gpu(*(
_output_shapes
:??
?
:gradients_3/decoder_1/layer_1/conv2d_transpose_grad/Conv2DConv2DCgradients_3/decoder_1/layer_1/BiasAdd_grad/tuple/control_dependencydecoder/layer_1/kernel/read*
strides
*
data_formatNHWC*
	dilations
*0
_output_shapes
:??????????*
paddingSAME*
explicit_paddings
 *
use_cudnn_on_gpu(*
T0
?
Dgradients_3/decoder_1/layer_1/conv2d_transpose_grad/tuple/group_depsNoOp;^gradients_3/decoder_1/layer_1/conv2d_transpose_grad/Conv2DI^gradients_3/decoder_1/layer_1/conv2d_transpose_grad/Conv2DBackpropFilter
?
Lgradients_3/decoder_1/layer_1/conv2d_transpose_grad/tuple/control_dependencyIdentityHgradients_3/decoder_1/layer_1/conv2d_transpose_grad/Conv2DBackpropFilterE^gradients_3/decoder_1/layer_1/conv2d_transpose_grad/tuple/group_deps*[
_classQ
OMloc:@gradients_3/decoder_1/layer_1/conv2d_transpose_grad/Conv2DBackpropFilter*(
_output_shapes
:??*
T0
?
Ngradients_3/decoder_1/layer_1/conv2d_transpose_grad/tuple/control_dependency_1Identity:gradients_3/decoder_1/layer_1/conv2d_transpose_grad/Conv2DE^gradients_3/decoder_1/layer_1/conv2d_transpose_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_3/decoder_1/layer_1/conv2d_transpose_grad/Conv2D*0
_output_shapes
:??????????
?
2gradients_3/decoder_1/LeakyRelu_grad/LeakyReluGradLeakyReluGradNgradients_3/decoder_1/layer_1/conv2d_transpose_grad/tuple/control_dependency_1,decoder_1/batch_normalization/FusedBatchNorm*0
_output_shapes
:??????????*
T0*
alpha%??L>
{
gradients_3/zeros_like_4	ZerosLike.decoder_1/batch_normalization/FusedBatchNorm:1*
T0*
_output_shapes	
:?
{
gradients_3/zeros_like_5	ZerosLike.decoder_1/batch_normalization/FusedBatchNorm:2*
T0*
_output_shapes	
:?
{
gradients_3/zeros_like_6	ZerosLike.decoder_1/batch_normalization/FusedBatchNorm:3*
_output_shapes	
:?*
T0
{
gradients_3/zeros_like_7	ZerosLike.decoder_1/batch_normalization/FusedBatchNorm:4*
_output_shapes	
:?*
T0
?
Pgradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGrad2gradients_3/decoder_1/LeakyRelu_grad/LeakyReluGraddecoder_1/layer_0/BiasAdd&decoder/batch_normalization/gamma/read.decoder_1/batch_normalization/FusedBatchNorm:3.decoder_1/batch_normalization/FusedBatchNorm:4*
is_training(*F
_output_shapes4
2:??????????:?:?: : *
epsilon%o?:*
T0*
data_formatNHWC
?
Ngradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/tuple/group_depsNoOpQ^gradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad
?
Vgradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependencyIdentityPgradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGradO^gradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*c
_classY
WUloc:@gradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*0
_output_shapes
:??????????
?
Xgradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_1IdentityRgradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad:1O^gradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*
_output_shapes	
:?*c
_classY
WUloc:@gradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad
?
Xgradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_2IdentityRgradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad:2O^gradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*c
_classY
WUloc:@gradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:?
?
Xgradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_3IdentityRgradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad:3O^gradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*
_output_shapes
: *c
_classY
WUloc:@gradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad
?
Xgradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_4IdentityRgradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad:4O^gradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
?
6gradients_3/decoder_1/layer_0/BiasAdd_grad/BiasAddGradBiasAddGradVgradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency*
T0*
_output_shapes	
:?*
data_formatNHWC
?
;gradients_3/decoder_1/layer_0/BiasAdd_grad/tuple/group_depsNoOpW^gradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency7^gradients_3/decoder_1/layer_0/BiasAdd_grad/BiasAddGrad
?
Cgradients_3/decoder_1/layer_0/BiasAdd_grad/tuple/control_dependencyIdentityVgradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency<^gradients_3/decoder_1/layer_0/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:??????????*
T0*c
_classY
WUloc:@gradients_3/decoder_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad
?
Egradients_3/decoder_1/layer_0/BiasAdd_grad/tuple/control_dependency_1Identity6gradients_3/decoder_1/layer_0/BiasAdd_grad/BiasAddGrad<^gradients_3/decoder_1/layer_0/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@gradients_3/decoder_1/layer_0/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?*
T0
?
9gradients_3/decoder_1/layer_0/conv2d_transpose_grad/ShapeConst*%
valueB"         d   *
_output_shapes
:*
dtype0
?
Hgradients_3/decoder_1/layer_0/conv2d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilterCgradients_3/decoder_1/layer_0/BiasAdd_grad/tuple/control_dependency9gradients_3/decoder_1/layer_0/conv2d_transpose_grad/Shapedecoder_1/Reshape*'
_output_shapes
:?d*
explicit_paddings
 *
use_cudnn_on_gpu(*
T0*
paddingVALID*
	dilations
*
data_formatNHWC*
strides

?
:gradients_3/decoder_1/layer_0/conv2d_transpose_grad/Conv2DConv2DCgradients_3/decoder_1/layer_0/BiasAdd_grad/tuple/control_dependencydecoder/layer_0/kernel/read*
use_cudnn_on_gpu(*
	dilations
*
strides
*
explicit_paddings
 *
data_formatNHWC*/
_output_shapes
:?????????d*
paddingVALID*
T0
?
Dgradients_3/decoder_1/layer_0/conv2d_transpose_grad/tuple/group_depsNoOp;^gradients_3/decoder_1/layer_0/conv2d_transpose_grad/Conv2DI^gradients_3/decoder_1/layer_0/conv2d_transpose_grad/Conv2DBackpropFilter
?
Lgradients_3/decoder_1/layer_0/conv2d_transpose_grad/tuple/control_dependencyIdentityHgradients_3/decoder_1/layer_0/conv2d_transpose_grad/Conv2DBackpropFilterE^gradients_3/decoder_1/layer_0/conv2d_transpose_grad/tuple/group_deps*'
_output_shapes
:?d*[
_classQ
OMloc:@gradients_3/decoder_1/layer_0/conv2d_transpose_grad/Conv2DBackpropFilter*
T0
?
Ngradients_3/decoder_1/layer_0/conv2d_transpose_grad/tuple/control_dependency_1Identity:gradients_3/decoder_1/layer_0/conv2d_transpose_grad/Conv2DE^gradients_3/decoder_1/layer_0/conv2d_transpose_grad/tuple/group_deps*
T0*/
_output_shapes
:?????????d*M
_classC
A?loc:@gradients_3/decoder_1/layer_0/conv2d_transpose_grad/Conv2D
?
(gradients_3/decoder_1/Reshape_grad/ShapeShape@MultivariateNormalDiag/sample/affine_linear_operator/forward/add*
_output_shapes
:*
T0*
out_type0
?
*gradients_3/decoder_1/Reshape_grad/ReshapeReshapeNgradients_3/decoder_1/layer_0/conv2d_transpose_grad/tuple/control_dependency_1(gradients_3/decoder_1/Reshape_grad/Shape*
Tshape0*
T0*'
_output_shapes
:?????????d
?
gradients_3/AddNAddN<gradients_3/diz_1/dense/MatMul_grad/tuple/control_dependency*gradients_3/decoder_1/Reshape_grad/Reshape*
N*=
_class3
1/loc:@gradients_3/diz_1/dense/MatMul_grad/MatMul*'
_output_shapes
:?????????d*
T0
?
Wgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/ShapeShapeZMultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul*
T0*
out_type0*
_output_shapes
:
?
Ygradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape_1ShapeHencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add*
out_type0*
_output_shapes
:*
T0
?
ggradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/BroadcastGradientArgsBroadcastGradientArgsWgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/ShapeYgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
Ugradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/SumSumgradients_3/AddNggradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
?
Ygradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/ReshapeReshapeUgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/SumWgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape*'
_output_shapes
:?????????d*
Tshape0*
T0
?
Wgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Sum_1Sumgradients_3/AddNigradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
[gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1ReshapeWgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Sum_1Ygradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape_1*
Tshape0*'
_output_shapes
:?????????d*
T0
?
bgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/group_depsNoOpZ^gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape\^gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1
?
jgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/control_dependencyIdentityYgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshapec^gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/group_deps*l
_classb
`^loc:@gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape*
T0*'
_output_shapes
:?????????d
?
lgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/control_dependency_1Identity[gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1c^gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/group_deps*n
_classd
b`loc:@gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1*
T0*'
_output_shapes
:?????????d
?
qgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/ShapeShapeadd*
_output_shapes
:*
T0*
out_type0
?
sgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape_1Shape%MultivariateNormalDiag/sample/Reshape*
T0*
_output_shapes
:*
out_type0
?
?gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/BroadcastGradientArgsBroadcastGradientArgsqgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shapesgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
ogradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/MulMuljgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/control_dependency%MultivariateNormalDiag/sample/Reshape*'
_output_shapes
:?????????d*
T0
?
ogradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/SumSumogradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Mul?gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
?
sgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/ReshapeReshapeogradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Sumqgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape*
T0*'
_output_shapes
:?????????d*
Tshape0
?
qgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Mul_1Muladdjgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/tuple/control_dependency*'
_output_shapes
:?????????d*
T0
?
qgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Sum_1Sumqgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Mul_1?gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
ugradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape_1Reshapeqgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Sum_1sgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????d
?
|gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/group_depsNoOpt^gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshapev^gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape_1
?
?gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/control_dependencyIdentitysgradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape}^gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/group_deps*?
_class|
zxloc:@gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape*'
_output_shapes
:?????????d*
T0
?
?gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/control_dependency_1Identityugradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape_1}^gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/group_deps*'
_output_shapes
:?????????d*
T0*?
_class~
|zloc:@gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape_1
n
gradients_3/add_grad/ShapeShapegen/dense_3/Softplus*
out_type0*
_output_shapes
:*
T0
_
gradients_3/add_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
?
*gradients_3/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/add_grad/Shapegradients_3/add_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_3/add_grad/SumSum?gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/control_dependency*gradients_3/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
gradients_3/add_grad/ReshapeReshapegradients_3/add_grad/Sumgradients_3/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????d
?
gradients_3/add_grad/Sum_1Sum?gradients_3/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/tuple/control_dependency,gradients_3/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
gradients_3/add_grad/Reshape_1Reshapegradients_3/add_grad/Sum_1gradients_3/add_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
m
%gradients_3/add_grad/tuple/group_depsNoOp^gradients_3/add_grad/Reshape^gradients_3/add_grad/Reshape_1
?
-gradients_3/add_grad/tuple/control_dependencyIdentitygradients_3/add_grad/Reshape&^gradients_3/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients_3/add_grad/Reshape*
T0*'
_output_shapes
:?????????d
?
/gradients_3/add_grad/tuple/control_dependency_1Identitygradients_3/add_grad/Reshape_1&^gradients_3/add_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_3/add_grad/Reshape_1*
_output_shapes
: 

-gradients_3/gen/dense_3/Softplus_grad/SigmoidSigmoidgen/dense_3/BiasAdd*'
_output_shapes
:?????????d*
T0
?
)gradients_3/gen/dense_3/Softplus_grad/mulMul-gradients_3/add_grad/tuple/control_dependency-gradients_3/gen/dense_3/Softplus_grad/Sigmoid*
T0*'
_output_shapes
:?????????d
?
0gradients_3/gen/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_3/gen/dense_3/Softplus_grad/mul*
data_formatNHWC*
T0*
_output_shapes
:d
?
5gradients_3/gen/dense_3/BiasAdd_grad/tuple/group_depsNoOp1^gradients_3/gen/dense_3/BiasAdd_grad/BiasAddGrad*^gradients_3/gen/dense_3/Softplus_grad/mul
?
=gradients_3/gen/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_3/gen/dense_3/Softplus_grad/mul6^gradients_3/gen/dense_3/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????d*<
_class2
0.loc:@gradients_3/gen/dense_3/Softplus_grad/mul
?
?gradients_3/gen/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients_3/gen/dense_3/BiasAdd_grad/BiasAddGrad6^gradients_3/gen/dense_3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_3/gen/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:d
?
*gradients_3/gen/dense_3/MatMul_grad/MatMulMatMul=gradients_3/gen/dense_3/BiasAdd_grad/tuple/control_dependencygen/dense_3/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:????????? 
?
,gradients_3/gen/dense_3/MatMul_grad/MatMul_1MatMulgen/dense_2/LeakyRelu=gradients_3/gen/dense_3/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

: d*
transpose_a(
?
4gradients_3/gen/dense_3/MatMul_grad/tuple/group_depsNoOp+^gradients_3/gen/dense_3/MatMul_grad/MatMul-^gradients_3/gen/dense_3/MatMul_grad/MatMul_1
?
<gradients_3/gen/dense_3/MatMul_grad/tuple/control_dependencyIdentity*gradients_3/gen/dense_3/MatMul_grad/MatMul5^gradients_3/gen/dense_3/MatMul_grad/tuple/group_deps*'
_output_shapes
:????????? *
T0*=
_class3
1/loc:@gradients_3/gen/dense_3/MatMul_grad/MatMul
?
>gradients_3/gen/dense_3/MatMul_grad/tuple/control_dependency_1Identity,gradients_3/gen/dense_3/MatMul_grad/MatMul_15^gradients_3/gen/dense_3/MatMul_grad/tuple/group_deps*
T0*
_output_shapes

: d*?
_class5
31loc:@gradients_3/gen/dense_3/MatMul_grad/MatMul_1
?
4gradients_3/gen/dense_2/LeakyRelu_grad/LeakyReluGradLeakyReluGrad<gradients_3/gen/dense_3/MatMul_grad/tuple/control_dependencygen/dense_2/BiasAdd*'
_output_shapes
:????????? *
alpha%??L>*
T0
?
0gradients_3/gen/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients_3/gen/dense_2/LeakyRelu_grad/LeakyReluGrad*
data_formatNHWC*
T0*
_output_shapes
: 
?
5gradients_3/gen/dense_2/BiasAdd_grad/tuple/group_depsNoOp1^gradients_3/gen/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_3/gen/dense_2/LeakyRelu_grad/LeakyReluGrad
?
=gradients_3/gen/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity4gradients_3/gen/dense_2/LeakyRelu_grad/LeakyReluGrad6^gradients_3/gen/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:????????? *
T0*G
_class=
;9loc:@gradients_3/gen/dense_2/LeakyRelu_grad/LeakyReluGrad
?
?gradients_3/gen/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients_3/gen/dense_2/BiasAdd_grad/BiasAddGrad6^gradients_3/gen/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
: *C
_class9
75loc:@gradients_3/gen/dense_2/BiasAdd_grad/BiasAddGrad*
T0
?
*gradients_3/gen/dense_2/MatMul_grad/MatMulMatMul=gradients_3/gen/dense_2/BiasAdd_grad/tuple/control_dependencygen/dense_2/kernel/read*
transpose_a( *
transpose_b(*'
_output_shapes
:????????? *
T0
?
,gradients_3/gen/dense_2/MatMul_grad/MatMul_1MatMulgen/dense_1/LeakyRelu=gradients_3/gen/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:  
?
4gradients_3/gen/dense_2/MatMul_grad/tuple/group_depsNoOp+^gradients_3/gen/dense_2/MatMul_grad/MatMul-^gradients_3/gen/dense_2/MatMul_grad/MatMul_1
?
<gradients_3/gen/dense_2/MatMul_grad/tuple/control_dependencyIdentity*gradients_3/gen/dense_2/MatMul_grad/MatMul5^gradients_3/gen/dense_2/MatMul_grad/tuple/group_deps*'
_output_shapes
:????????? *=
_class3
1/loc:@gradients_3/gen/dense_2/MatMul_grad/MatMul*
T0
?
>gradients_3/gen/dense_2/MatMul_grad/tuple/control_dependency_1Identity,gradients_3/gen/dense_2/MatMul_grad/MatMul_15^gradients_3/gen/dense_2/MatMul_grad/tuple/group_deps*
T0*
_output_shapes

:  *?
_class5
31loc:@gradients_3/gen/dense_2/MatMul_grad/MatMul_1
?
4gradients_3/gen/dense_1/LeakyRelu_grad/LeakyReluGradLeakyReluGrad<gradients_3/gen/dense_2/MatMul_grad/tuple/control_dependencygen/dense_1/BiasAdd*
T0*
alpha%??L>*'
_output_shapes
:????????? 
?
0gradients_3/gen/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients_3/gen/dense_1/LeakyRelu_grad/LeakyReluGrad*
_output_shapes
: *
T0*
data_formatNHWC
?
5gradients_3/gen/dense_1/BiasAdd_grad/tuple/group_depsNoOp1^gradients_3/gen/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_3/gen/dense_1/LeakyRelu_grad/LeakyReluGrad
?
=gradients_3/gen/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity4gradients_3/gen/dense_1/LeakyRelu_grad/LeakyReluGrad6^gradients_3/gen/dense_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:????????? *G
_class=
;9loc:@gradients_3/gen/dense_1/LeakyRelu_grad/LeakyReluGrad*
T0
?
?gradients_3/gen/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients_3/gen/dense_1/BiasAdd_grad/BiasAddGrad6^gradients_3/gen/dense_1/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@gradients_3/gen/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
?
*gradients_3/gen/dense_1/MatMul_grad/MatMulMatMul=gradients_3/gen/dense_1/BiasAdd_grad/tuple/control_dependencygen/dense_1/kernel/read*
T0*
transpose_b(*
transpose_a( *'
_output_shapes
:????????? 
?
,gradients_3/gen/dense_1/MatMul_grad/MatMul_1MatMulgen/dense/LeakyRelu=gradients_3/gen/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:  *
T0*
transpose_a(
?
4gradients_3/gen/dense_1/MatMul_grad/tuple/group_depsNoOp+^gradients_3/gen/dense_1/MatMul_grad/MatMul-^gradients_3/gen/dense_1/MatMul_grad/MatMul_1
?
<gradients_3/gen/dense_1/MatMul_grad/tuple/control_dependencyIdentity*gradients_3/gen/dense_1/MatMul_grad/MatMul5^gradients_3/gen/dense_1/MatMul_grad/tuple/group_deps*'
_output_shapes
:????????? *=
_class3
1/loc:@gradients_3/gen/dense_1/MatMul_grad/MatMul*
T0
?
>gradients_3/gen/dense_1/MatMul_grad/tuple/control_dependency_1Identity,gradients_3/gen/dense_1/MatMul_grad/MatMul_15^gradients_3/gen/dense_1/MatMul_grad/tuple/group_deps*?
_class5
31loc:@gradients_3/gen/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:  *
T0
?
2gradients_3/gen/dense/LeakyRelu_grad/LeakyReluGradLeakyReluGrad<gradients_3/gen/dense_1/MatMul_grad/tuple/control_dependencygen/dense/BiasAdd*'
_output_shapes
:????????? *
alpha%??L>*
T0
?
.gradients_3/gen/dense/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients_3/gen/dense/LeakyRelu_grad/LeakyReluGrad*
data_formatNHWC*
_output_shapes
: *
T0
?
3gradients_3/gen/dense/BiasAdd_grad/tuple/group_depsNoOp/^gradients_3/gen/dense/BiasAdd_grad/BiasAddGrad3^gradients_3/gen/dense/LeakyRelu_grad/LeakyReluGrad
?
;gradients_3/gen/dense/BiasAdd_grad/tuple/control_dependencyIdentity2gradients_3/gen/dense/LeakyRelu_grad/LeakyReluGrad4^gradients_3/gen/dense/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_3/gen/dense/LeakyRelu_grad/LeakyReluGrad*'
_output_shapes
:????????? 
?
=gradients_3/gen/dense/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_3/gen/dense/BiasAdd_grad/BiasAddGrad4^gradients_3/gen/dense/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients_3/gen/dense/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
?
(gradients_3/gen/dense/MatMul_grad/MatMulMatMul;gradients_3/gen/dense/BiasAdd_grad/tuple/control_dependencygen/dense/kernel/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:?????????f
?
*gradients_3/gen/dense/MatMul_grad/MatMul_1MatMul
gen/concat;gradients_3/gen/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:f *
transpose_a(*
transpose_b( *
T0
?
2gradients_3/gen/dense/MatMul_grad/tuple/group_depsNoOp)^gradients_3/gen/dense/MatMul_grad/MatMul+^gradients_3/gen/dense/MatMul_grad/MatMul_1
?
:gradients_3/gen/dense/MatMul_grad/tuple/control_dependencyIdentity(gradients_3/gen/dense/MatMul_grad/MatMul3^gradients_3/gen/dense/MatMul_grad/tuple/group_deps*;
_class1
/-loc:@gradients_3/gen/dense/MatMul_grad/MatMul*
T0*'
_output_shapes
:?????????f
?
<gradients_3/gen/dense/MatMul_grad/tuple/control_dependency_1Identity*gradients_3/gen/dense/MatMul_grad/MatMul_13^gradients_3/gen/dense/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_3/gen/dense/MatMul_grad/MatMul_1*
_output_shapes

:f 
?
9gen/dense/kernel/RMSProp/Initializer/ones/shape_as_tensorConst*#
_class
loc:@gen/dense/kernel*
dtype0*
valueB"f       *
_output_shapes
:
?
/gen/dense/kernel/RMSProp/Initializer/ones/ConstConst*
valueB
 *  ??*#
_class
loc:@gen/dense/kernel*
dtype0*
_output_shapes
: 
?
)gen/dense/kernel/RMSProp/Initializer/onesFill9gen/dense/kernel/RMSProp/Initializer/ones/shape_as_tensor/gen/dense/kernel/RMSProp/Initializer/ones/Const*#
_class
loc:@gen/dense/kernel*

index_type0*
_output_shapes

:f *
T0
?
gen/dense/kernel/RMSProp
VariableV2*
_output_shapes

:f *
	container *
dtype0*
shared_name *
shape
:f *#
_class
loc:@gen/dense/kernel
?
gen/dense/kernel/RMSProp/AssignAssigngen/dense/kernel/RMSProp)gen/dense/kernel/RMSProp/Initializer/ones*
validate_shape(*
T0*#
_class
loc:@gen/dense/kernel*
_output_shapes

:f *
use_locking(
?
gen/dense/kernel/RMSProp/readIdentitygen/dense/kernel/RMSProp*#
_class
loc:@gen/dense/kernel*
_output_shapes

:f *
T0
?
<gen/dense/kernel/RMSProp_1/Initializer/zeros/shape_as_tensorConst*
valueB"f       *#
_class
loc:@gen/dense/kernel*
dtype0*
_output_shapes
:
?
2gen/dense/kernel/RMSProp_1/Initializer/zeros/ConstConst*
valueB
 *    *#
_class
loc:@gen/dense/kernel*
_output_shapes
: *
dtype0
?
,gen/dense/kernel/RMSProp_1/Initializer/zerosFill<gen/dense/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor2gen/dense/kernel/RMSProp_1/Initializer/zeros/Const*#
_class
loc:@gen/dense/kernel*
T0*
_output_shapes

:f *

index_type0
?
gen/dense/kernel/RMSProp_1
VariableV2*
	container *
shared_name *#
_class
loc:@gen/dense/kernel*
dtype0*
shape
:f *
_output_shapes

:f 
?
!gen/dense/kernel/RMSProp_1/AssignAssigngen/dense/kernel/RMSProp_1,gen/dense/kernel/RMSProp_1/Initializer/zeros*
_output_shapes

:f *
T0*
use_locking(*
validate_shape(*#
_class
loc:@gen/dense/kernel
?
gen/dense/kernel/RMSProp_1/readIdentitygen/dense/kernel/RMSProp_1*
_output_shapes

:f *#
_class
loc:@gen/dense/kernel*
T0
?
'gen/dense/bias/RMSProp/Initializer/onesConst*!
_class
loc:@gen/dense/bias*
valueB *  ??*
dtype0*
_output_shapes
: 
?
gen/dense/bias/RMSProp
VariableV2*
	container *
dtype0*
shape: *
_output_shapes
: *!
_class
loc:@gen/dense/bias*
shared_name 
?
gen/dense/bias/RMSProp/AssignAssigngen/dense/bias/RMSProp'gen/dense/bias/RMSProp/Initializer/ones*!
_class
loc:@gen/dense/bias*
_output_shapes
: *
T0*
use_locking(*
validate_shape(
?
gen/dense/bias/RMSProp/readIdentitygen/dense/bias/RMSProp*!
_class
loc:@gen/dense/bias*
T0*
_output_shapes
: 
?
*gen/dense/bias/RMSProp_1/Initializer/zerosConst*
dtype0*!
_class
loc:@gen/dense/bias*
valueB *    *
_output_shapes
: 
?
gen/dense/bias/RMSProp_1
VariableV2*
shared_name *
shape: *
	container *
_output_shapes
: *!
_class
loc:@gen/dense/bias*
dtype0
?
gen/dense/bias/RMSProp_1/AssignAssigngen/dense/bias/RMSProp_1*gen/dense/bias/RMSProp_1/Initializer/zeros*
validate_shape(*
_output_shapes
: *
T0*
use_locking(*!
_class
loc:@gen/dense/bias
?
gen/dense/bias/RMSProp_1/readIdentitygen/dense/bias/RMSProp_1*!
_class
loc:@gen/dense/bias*
T0*
_output_shapes
: 
?
;gen/dense_1/kernel/RMSProp/Initializer/ones/shape_as_tensorConst*
dtype0*
valueB"        *%
_class
loc:@gen/dense_1/kernel*
_output_shapes
:
?
1gen/dense_1/kernel/RMSProp/Initializer/ones/ConstConst*%
_class
loc:@gen/dense_1/kernel*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
+gen/dense_1/kernel/RMSProp/Initializer/onesFill;gen/dense_1/kernel/RMSProp/Initializer/ones/shape_as_tensor1gen/dense_1/kernel/RMSProp/Initializer/ones/Const*
T0*%
_class
loc:@gen/dense_1/kernel*

index_type0*
_output_shapes

:  
?
gen/dense_1/kernel/RMSProp
VariableV2*
shared_name *
dtype0*
shape
:  *%
_class
loc:@gen/dense_1/kernel*
_output_shapes

:  *
	container 
?
!gen/dense_1/kernel/RMSProp/AssignAssigngen/dense_1/kernel/RMSProp+gen/dense_1/kernel/RMSProp/Initializer/ones*
T0*%
_class
loc:@gen/dense_1/kernel*
_output_shapes

:  *
use_locking(*
validate_shape(
?
gen/dense_1/kernel/RMSProp/readIdentitygen/dense_1/kernel/RMSProp*
_output_shapes

:  *
T0*%
_class
loc:@gen/dense_1/kernel
?
>gen/dense_1/kernel/RMSProp_1/Initializer/zeros/shape_as_tensorConst*%
_class
loc:@gen/dense_1/kernel*
valueB"        *
dtype0*
_output_shapes
:
?
4gen/dense_1/kernel/RMSProp_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *%
_class
loc:@gen/dense_1/kernel*
dtype0
?
.gen/dense_1/kernel/RMSProp_1/Initializer/zerosFill>gen/dense_1/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor4gen/dense_1/kernel/RMSProp_1/Initializer/zeros/Const*
T0*%
_class
loc:@gen/dense_1/kernel*

index_type0*
_output_shapes

:  
?
gen/dense_1/kernel/RMSProp_1
VariableV2*
dtype0*
shape
:  *
_output_shapes

:  *
shared_name *%
_class
loc:@gen/dense_1/kernel*
	container 
?
#gen/dense_1/kernel/RMSProp_1/AssignAssigngen/dense_1/kernel/RMSProp_1.gen/dense_1/kernel/RMSProp_1/Initializer/zeros*%
_class
loc:@gen/dense_1/kernel*
T0*
_output_shapes

:  *
validate_shape(*
use_locking(
?
!gen/dense_1/kernel/RMSProp_1/readIdentitygen/dense_1/kernel/RMSProp_1*%
_class
loc:@gen/dense_1/kernel*
_output_shapes

:  *
T0
?
)gen/dense_1/bias/RMSProp/Initializer/onesConst*
_output_shapes
: *#
_class
loc:@gen/dense_1/bias*
valueB *  ??*
dtype0
?
gen/dense_1/bias/RMSProp
VariableV2*
shared_name *
	container *#
_class
loc:@gen/dense_1/bias*
_output_shapes
: *
dtype0*
shape: 
?
gen/dense_1/bias/RMSProp/AssignAssigngen/dense_1/bias/RMSProp)gen/dense_1/bias/RMSProp/Initializer/ones*
use_locking(*
_output_shapes
: *
validate_shape(*#
_class
loc:@gen/dense_1/bias*
T0
?
gen/dense_1/bias/RMSProp/readIdentitygen/dense_1/bias/RMSProp*
_output_shapes
: *#
_class
loc:@gen/dense_1/bias*
T0
?
,gen/dense_1/bias/RMSProp_1/Initializer/zerosConst*
valueB *    *
dtype0*
_output_shapes
: *#
_class
loc:@gen/dense_1/bias
?
gen/dense_1/bias/RMSProp_1
VariableV2*
	container *#
_class
loc:@gen/dense_1/bias*
shared_name *
dtype0*
shape: *
_output_shapes
: 
?
!gen/dense_1/bias/RMSProp_1/AssignAssigngen/dense_1/bias/RMSProp_1,gen/dense_1/bias/RMSProp_1/Initializer/zeros*
_output_shapes
: *#
_class
loc:@gen/dense_1/bias*
use_locking(*
T0*
validate_shape(
?
gen/dense_1/bias/RMSProp_1/readIdentitygen/dense_1/bias/RMSProp_1*#
_class
loc:@gen/dense_1/bias*
_output_shapes
: *
T0
?
;gen/dense_2/kernel/RMSProp/Initializer/ones/shape_as_tensorConst*%
_class
loc:@gen/dense_2/kernel*
valueB"        *
dtype0*
_output_shapes
:
?
1gen/dense_2/kernel/RMSProp/Initializer/ones/ConstConst*
_output_shapes
: *
valueB
 *  ??*
dtype0*%
_class
loc:@gen/dense_2/kernel
?
+gen/dense_2/kernel/RMSProp/Initializer/onesFill;gen/dense_2/kernel/RMSProp/Initializer/ones/shape_as_tensor1gen/dense_2/kernel/RMSProp/Initializer/ones/Const*%
_class
loc:@gen/dense_2/kernel*

index_type0*
T0*
_output_shapes

:  
?
gen/dense_2/kernel/RMSProp
VariableV2*
shared_name *
dtype0*%
_class
loc:@gen/dense_2/kernel*
shape
:  *
_output_shapes

:  *
	container 
?
!gen/dense_2/kernel/RMSProp/AssignAssigngen/dense_2/kernel/RMSProp+gen/dense_2/kernel/RMSProp/Initializer/ones*
_output_shapes

:  *%
_class
loc:@gen/dense_2/kernel*
validate_shape(*
use_locking(*
T0
?
gen/dense_2/kernel/RMSProp/readIdentitygen/dense_2/kernel/RMSProp*
_output_shapes

:  *%
_class
loc:@gen/dense_2/kernel*
T0
?
>gen/dense_2/kernel/RMSProp_1/Initializer/zeros/shape_as_tensorConst*
valueB"        *%
_class
loc:@gen/dense_2/kernel*
dtype0*
_output_shapes
:
?
4gen/dense_2/kernel/RMSProp_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*%
_class
loc:@gen/dense_2/kernel*
_output_shapes
: 
?
.gen/dense_2/kernel/RMSProp_1/Initializer/zerosFill>gen/dense_2/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor4gen/dense_2/kernel/RMSProp_1/Initializer/zeros/Const*
T0*

index_type0*%
_class
loc:@gen/dense_2/kernel*
_output_shapes

:  
?
gen/dense_2/kernel/RMSProp_1
VariableV2*
shared_name *
shape
:  *
	container *
dtype0*
_output_shapes

:  *%
_class
loc:@gen/dense_2/kernel
?
#gen/dense_2/kernel/RMSProp_1/AssignAssigngen/dense_2/kernel/RMSProp_1.gen/dense_2/kernel/RMSProp_1/Initializer/zeros*
validate_shape(*
T0*%
_class
loc:@gen/dense_2/kernel*
use_locking(*
_output_shapes

:  
?
!gen/dense_2/kernel/RMSProp_1/readIdentitygen/dense_2/kernel/RMSProp_1*
T0*%
_class
loc:@gen/dense_2/kernel*
_output_shapes

:  
?
)gen/dense_2/bias/RMSProp/Initializer/onesConst*
_output_shapes
: *#
_class
loc:@gen/dense_2/bias*
valueB *  ??*
dtype0
?
gen/dense_2/bias/RMSProp
VariableV2*
_output_shapes
: *
shape: *
	container *
shared_name *#
_class
loc:@gen/dense_2/bias*
dtype0
?
gen/dense_2/bias/RMSProp/AssignAssigngen/dense_2/bias/RMSProp)gen/dense_2/bias/RMSProp/Initializer/ones*
use_locking(*
T0*
validate_shape(*
_output_shapes
: *#
_class
loc:@gen/dense_2/bias
?
gen/dense_2/bias/RMSProp/readIdentitygen/dense_2/bias/RMSProp*
_output_shapes
: *#
_class
loc:@gen/dense_2/bias*
T0
?
,gen/dense_2/bias/RMSProp_1/Initializer/zerosConst*#
_class
loc:@gen/dense_2/bias*
_output_shapes
: *
dtype0*
valueB *    
?
gen/dense_2/bias/RMSProp_1
VariableV2*
	container *#
_class
loc:@gen/dense_2/bias*
dtype0*
shape: *
_output_shapes
: *
shared_name 
?
!gen/dense_2/bias/RMSProp_1/AssignAssigngen/dense_2/bias/RMSProp_1,gen/dense_2/bias/RMSProp_1/Initializer/zeros*
validate_shape(*#
_class
loc:@gen/dense_2/bias*
use_locking(*
T0*
_output_shapes
: 
?
gen/dense_2/bias/RMSProp_1/readIdentitygen/dense_2/bias/RMSProp_1*
_output_shapes
: *#
_class
loc:@gen/dense_2/bias*
T0
?
;gen/dense_3/kernel/RMSProp/Initializer/ones/shape_as_tensorConst*
dtype0*
valueB"    d   *%
_class
loc:@gen/dense_3/kernel*
_output_shapes
:
?
1gen/dense_3/kernel/RMSProp/Initializer/ones/ConstConst*%
_class
loc:@gen/dense_3/kernel*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
+gen/dense_3/kernel/RMSProp/Initializer/onesFill;gen/dense_3/kernel/RMSProp/Initializer/ones/shape_as_tensor1gen/dense_3/kernel/RMSProp/Initializer/ones/Const*%
_class
loc:@gen/dense_3/kernel*
_output_shapes

: d*
T0*

index_type0
?
gen/dense_3/kernel/RMSProp
VariableV2*
	container *
shared_name *
_output_shapes

: d*%
_class
loc:@gen/dense_3/kernel*
dtype0*
shape
: d
?
!gen/dense_3/kernel/RMSProp/AssignAssigngen/dense_3/kernel/RMSProp+gen/dense_3/kernel/RMSProp/Initializer/ones*
_output_shapes

: d*
T0*
use_locking(*%
_class
loc:@gen/dense_3/kernel*
validate_shape(
?
gen/dense_3/kernel/RMSProp/readIdentitygen/dense_3/kernel/RMSProp*
T0*%
_class
loc:@gen/dense_3/kernel*
_output_shapes

: d
?
>gen/dense_3/kernel/RMSProp_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"    d   *%
_class
loc:@gen/dense_3/kernel
?
4gen/dense_3/kernel/RMSProp_1/Initializer/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    *%
_class
loc:@gen/dense_3/kernel
?
.gen/dense_3/kernel/RMSProp_1/Initializer/zerosFill>gen/dense_3/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor4gen/dense_3/kernel/RMSProp_1/Initializer/zeros/Const*
_output_shapes

: d*%
_class
loc:@gen/dense_3/kernel*
T0*

index_type0
?
gen/dense_3/kernel/RMSProp_1
VariableV2*
_output_shapes

: d*
dtype0*%
_class
loc:@gen/dense_3/kernel*
	container *
shape
: d*
shared_name 
?
#gen/dense_3/kernel/RMSProp_1/AssignAssigngen/dense_3/kernel/RMSProp_1.gen/dense_3/kernel/RMSProp_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*%
_class
loc:@gen/dense_3/kernel*
_output_shapes

: d
?
!gen/dense_3/kernel/RMSProp_1/readIdentitygen/dense_3/kernel/RMSProp_1*
T0*%
_class
loc:@gen/dense_3/kernel*
_output_shapes

: d
?
)gen/dense_3/bias/RMSProp/Initializer/onesConst*#
_class
loc:@gen/dense_3/bias*
valueBd*  ??*
_output_shapes
:d*
dtype0
?
gen/dense_3/bias/RMSProp
VariableV2*
shared_name *
dtype0*#
_class
loc:@gen/dense_3/bias*
	container *
_output_shapes
:d*
shape:d
?
gen/dense_3/bias/RMSProp/AssignAssigngen/dense_3/bias/RMSProp)gen/dense_3/bias/RMSProp/Initializer/ones*
T0*
validate_shape(*
use_locking(*#
_class
loc:@gen/dense_3/bias*
_output_shapes
:d
?
gen/dense_3/bias/RMSProp/readIdentitygen/dense_3/bias/RMSProp*
T0*#
_class
loc:@gen/dense_3/bias*
_output_shapes
:d
?
,gen/dense_3/bias/RMSProp_1/Initializer/zerosConst*
_output_shapes
:d*#
_class
loc:@gen/dense_3/bias*
valueBd*    *
dtype0
?
gen/dense_3/bias/RMSProp_1
VariableV2*
shared_name *
dtype0*
shape:d*
_output_shapes
:d*#
_class
loc:@gen/dense_3/bias*
	container 
?
!gen/dense_3/bias/RMSProp_1/AssignAssigngen/dense_3/bias/RMSProp_1,gen/dense_3/bias/RMSProp_1/Initializer/zeros*
use_locking(*#
_class
loc:@gen/dense_3/bias*
validate_shape(*
_output_shapes
:d*
T0
?
gen/dense_3/bias/RMSProp_1/readIdentitygen/dense_3/bias/RMSProp_1*
T0*#
_class
loc:@gen/dense_3/bias*
_output_shapes
:d
\
RMSProp_2/learning_rateConst*
dtype0*
valueB
 *??8*
_output_shapes
: 
T
RMSProp_2/decayConst*
dtype0*
_output_shapes
: *
valueB
 *fff?
W
RMSProp_2/momentumConst*
_output_shapes
: *
dtype0*
valueB
 *    
V
RMSProp_2/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *???.
?
.RMSProp_2/update_gen/dense/kernel/ApplyRMSPropApplyRMSPropgen/dense/kernelgen/dense/kernel/RMSPropgen/dense/kernel/RMSProp_1RMSProp_2/learning_rateRMSProp_2/decayRMSProp_2/momentumRMSProp_2/epsilon<gradients_3/gen/dense/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes

:f *#
_class
loc:@gen/dense/kernel
?
,RMSProp_2/update_gen/dense/bias/ApplyRMSPropApplyRMSPropgen/dense/biasgen/dense/bias/RMSPropgen/dense/bias/RMSProp_1RMSProp_2/learning_rateRMSProp_2/decayRMSProp_2/momentumRMSProp_2/epsilon=gradients_3/gen/dense/BiasAdd_grad/tuple/control_dependency_1*!
_class
loc:@gen/dense/bias*
use_locking( *
_output_shapes
: *
T0
?
0RMSProp_2/update_gen/dense_1/kernel/ApplyRMSPropApplyRMSPropgen/dense_1/kernelgen/dense_1/kernel/RMSPropgen/dense_1/kernel/RMSProp_1RMSProp_2/learning_rateRMSProp_2/decayRMSProp_2/momentumRMSProp_2/epsilon>gradients_3/gen/dense_1/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:  *
T0*
use_locking( *%
_class
loc:@gen/dense_1/kernel
?
.RMSProp_2/update_gen/dense_1/bias/ApplyRMSPropApplyRMSPropgen/dense_1/biasgen/dense_1/bias/RMSPropgen/dense_1/bias/RMSProp_1RMSProp_2/learning_rateRMSProp_2/decayRMSProp_2/momentumRMSProp_2/epsilon?gradients_3/gen/dense_1/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
: *
T0*#
_class
loc:@gen/dense_1/bias*
use_locking( 
?
0RMSProp_2/update_gen/dense_2/kernel/ApplyRMSPropApplyRMSPropgen/dense_2/kernelgen/dense_2/kernel/RMSPropgen/dense_2/kernel/RMSProp_1RMSProp_2/learning_rateRMSProp_2/decayRMSProp_2/momentumRMSProp_2/epsilon>gradients_3/gen/dense_2/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:  *
use_locking( *%
_class
loc:@gen/dense_2/kernel*
T0
?
.RMSProp_2/update_gen/dense_2/bias/ApplyRMSPropApplyRMSPropgen/dense_2/biasgen/dense_2/bias/RMSPropgen/dense_2/bias/RMSProp_1RMSProp_2/learning_rateRMSProp_2/decayRMSProp_2/momentumRMSProp_2/epsilon?gradients_3/gen/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
_output_shapes
: *
T0*#
_class
loc:@gen/dense_2/bias
?
0RMSProp_2/update_gen/dense_3/kernel/ApplyRMSPropApplyRMSPropgen/dense_3/kernelgen/dense_3/kernel/RMSPropgen/dense_3/kernel/RMSProp_1RMSProp_2/learning_rateRMSProp_2/decayRMSProp_2/momentumRMSProp_2/epsilon>gradients_3/gen/dense_3/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes

: d*%
_class
loc:@gen/dense_3/kernel
?
.RMSProp_2/update_gen/dense_3/bias/ApplyRMSPropApplyRMSPropgen/dense_3/biasgen/dense_3/bias/RMSPropgen/dense_3/bias/RMSProp_1RMSProp_2/learning_rateRMSProp_2/decayRMSProp_2/momentumRMSProp_2/epsilon?gradients_3/gen/dense_3/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
_output_shapes
:d*
T0*#
_class
loc:@gen/dense_3/bias
?
	RMSProp_2NoOp-^RMSProp_2/update_gen/dense/bias/ApplyRMSProp/^RMSProp_2/update_gen/dense/kernel/ApplyRMSProp/^RMSProp_2/update_gen/dense_1/bias/ApplyRMSProp1^RMSProp_2/update_gen/dense_1/kernel/ApplyRMSProp/^RMSProp_2/update_gen/dense_2/bias/ApplyRMSProp1^RMSProp_2/update_gen/dense_2/kernel/ApplyRMSProp/^RMSProp_2/update_gen/dense_3/bias/ApplyRMSProp1^RMSProp_2/update_gen/dense_3/kernel/ApplyRMSProp


group_depsNoOp^Adam
C
group_deps_1NoOp^RMSProp
^RMSProp_1
^RMSProp_2^group_deps
k
disc/Reshape/shapeConst*%
valueB"????         *
dtype0*
_output_shapes
:
v
disc/ReshapeReshapeXdisc/Reshape/shape*
Tshape0*/
_output_shapes
:?????????*
T0
?
5disc/conv2d/kernel/Initializer/truncated_normal/shapeConst*%
valueB"            *
_output_shapes
:*%
_class
loc:@disc/conv2d/kernel*
dtype0
?
4disc/conv2d/kernel/Initializer/truncated_normal/meanConst*%
_class
loc:@disc/conv2d/kernel*
_output_shapes
: *
valueB
 *    *
dtype0
?
6disc/conv2d/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *??h>*%
_class
loc:@disc/conv2d/kernel
?
?disc/conv2d/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal5disc/conv2d/kernel/Initializer/truncated_normal/shape*
dtype0*
T0*

seed *&
_output_shapes
:*%
_class
loc:@disc/conv2d/kernel*
seed2 
?
3disc/conv2d/kernel/Initializer/truncated_normal/mulMul?disc/conv2d/kernel/Initializer/truncated_normal/TruncatedNormal6disc/conv2d/kernel/Initializer/truncated_normal/stddev*
T0*&
_output_shapes
:*%
_class
loc:@disc/conv2d/kernel
?
/disc/conv2d/kernel/Initializer/truncated_normalAdd3disc/conv2d/kernel/Initializer/truncated_normal/mul4disc/conv2d/kernel/Initializer/truncated_normal/mean*&
_output_shapes
:*%
_class
loc:@disc/conv2d/kernel*
T0
?
disc/conv2d/kernel
VariableV2*
	container *&
_output_shapes
:*
shape:*
dtype0*
shared_name *%
_class
loc:@disc/conv2d/kernel
?
disc/conv2d/kernel/AssignAssigndisc/conv2d/kernel/disc/conv2d/kernel/Initializer/truncated_normal*
validate_shape(*
T0*
use_locking(*&
_output_shapes
:*%
_class
loc:@disc/conv2d/kernel
?
disc/conv2d/kernel/readIdentitydisc/conv2d/kernel*&
_output_shapes
:*%
_class
loc:@disc/conv2d/kernel*
T0
x
3disc/conv2d/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *
valueB
 *o?:
x
4disc/conv2d/kernel/Regularizer/l2_regularizer/L2LossL2Lossdisc/conv2d/kernel/read*
_output_shapes
: *
T0
?
-disc/conv2d/kernel/Regularizer/l2_regularizerMul3disc/conv2d/kernel/Regularizer/l2_regularizer/scale4disc/conv2d/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
"disc/conv2d/bias/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*#
_class
loc:@disc/conv2d/bias
?
disc/conv2d/bias
VariableV2*#
_class
loc:@disc/conv2d/bias*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
?
disc/conv2d/bias/AssignAssigndisc/conv2d/bias"disc/conv2d/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*#
_class
loc:@disc/conv2d/bias*
use_locking(*
T0
}
disc/conv2d/bias/readIdentitydisc/conv2d/bias*
T0*#
_class
loc:@disc/conv2d/bias*
_output_shapes
:
j
disc/conv2d/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
?
disc/conv2d/Conv2DConv2Ddisc/Reshapedisc/conv2d/kernel/read*
T0*
data_formatNHWC*
strides
*
paddingVALID*
explicit_paddings
 *
use_cudnn_on_gpu(*
	dilations
*/
_output_shapes
:?????????
?
disc/conv2d/BiasAddBiasAdddisc/conv2d/Conv2Ddisc/conv2d/bias/read*
T0*/
_output_shapes
:?????????*
data_formatNHWC
g
disc/conv2d/ReluReludisc/conv2d/BiasAdd*/
_output_shapes
:?????????*
T0
?
disc/MaxPoolMaxPooldisc/conv2d/Relu*
strides
*/
_output_shapes
:?????????*
paddingSAME*
T0*
ksize
*
data_formatNHWC
?
7disc/conv2d_1/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*'
_class
loc:@disc/conv2d_1/kernel*%
valueB"         2   *
dtype0
?
6disc/conv2d_1/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *'
_class
loc:@disc/conv2d_1/kernel*
valueB
 *    *
dtype0
?
8disc/conv2d_1/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *'
_class
loc:@disc/conv2d_1/kernel*
valueB
 *?P=*
dtype0
?
Adisc/conv2d_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal7disc/conv2d_1/kernel/Initializer/truncated_normal/shape*'
_class
loc:@disc/conv2d_1/kernel*
dtype0*

seed *
T0*
seed2 *&
_output_shapes
:2
?
5disc/conv2d_1/kernel/Initializer/truncated_normal/mulMulAdisc/conv2d_1/kernel/Initializer/truncated_normal/TruncatedNormal8disc/conv2d_1/kernel/Initializer/truncated_normal/stddev*&
_output_shapes
:2*
T0*'
_class
loc:@disc/conv2d_1/kernel
?
1disc/conv2d_1/kernel/Initializer/truncated_normalAdd5disc/conv2d_1/kernel/Initializer/truncated_normal/mul6disc/conv2d_1/kernel/Initializer/truncated_normal/mean*&
_output_shapes
:2*
T0*'
_class
loc:@disc/conv2d_1/kernel
?
disc/conv2d_1/kernel
VariableV2*
shape:2*'
_class
loc:@disc/conv2d_1/kernel*&
_output_shapes
:2*
dtype0*
shared_name *
	container 
?
disc/conv2d_1/kernel/AssignAssigndisc/conv2d_1/kernel1disc/conv2d_1/kernel/Initializer/truncated_normal*'
_class
loc:@disc/conv2d_1/kernel*&
_output_shapes
:2*
T0*
use_locking(*
validate_shape(
?
disc/conv2d_1/kernel/readIdentitydisc/conv2d_1/kernel*&
_output_shapes
:2*
T0*'
_class
loc:@disc/conv2d_1/kernel
z
5disc/conv2d_1/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
valueB
 *o?:*
dtype0
|
6disc/conv2d_1/kernel/Regularizer/l2_regularizer/L2LossL2Lossdisc/conv2d_1/kernel/read*
T0*
_output_shapes
: 
?
/disc/conv2d_1/kernel/Regularizer/l2_regularizerMul5disc/conv2d_1/kernel/Regularizer/l2_regularizer/scale6disc/conv2d_1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
$disc/conv2d_1/bias/Initializer/zerosConst*%
_class
loc:@disc/conv2d_1/bias*
_output_shapes
:2*
valueB2*    *
dtype0
?
disc/conv2d_1/bias
VariableV2*
shape:2*
_output_shapes
:2*
shared_name *%
_class
loc:@disc/conv2d_1/bias*
	container *
dtype0
?
disc/conv2d_1/bias/AssignAssigndisc/conv2d_1/bias$disc/conv2d_1/bias/Initializer/zeros*
validate_shape(*
use_locking(*
T0*%
_class
loc:@disc/conv2d_1/bias*
_output_shapes
:2
?
disc/conv2d_1/bias/readIdentitydisc/conv2d_1/bias*%
_class
loc:@disc/conv2d_1/bias*
T0*
_output_shapes
:2
l
disc/conv2d_1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
disc/conv2d_1/Conv2DConv2Ddisc/MaxPooldisc/conv2d_1/kernel/read*
T0*
explicit_paddings
 */
_output_shapes
:?????????2*
strides
*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
	dilations

?
disc/conv2d_1/BiasAddBiasAdddisc/conv2d_1/Conv2Ddisc/conv2d_1/bias/read*/
_output_shapes
:?????????2*
data_formatNHWC*
T0
k
disc/conv2d_1/ReluReludisc/conv2d_1/BiasAdd*/
_output_shapes
:?????????2*
T0
?
disc/MaxPool_1MaxPooldisc/conv2d_1/Relu*/
_output_shapes
:?????????2*
ksize
*
data_formatNHWC*
T0*
paddingSAME*
strides

`
disc/flatten/ShapeShapedisc/MaxPool_1*
T0*
out_type0*
_output_shapes
:
j
 disc/flatten/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
l
"disc/flatten/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
l
"disc/flatten/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
disc/flatten/strided_sliceStridedSlicedisc/flatten/Shape disc/flatten/strided_slice/stack"disc/flatten/strided_slice/stack_1"disc/flatten/strided_slice/stack_2*
Index0*
_output_shapes
: *
shrink_axis_mask*
end_mask *

begin_mask *
ellipsis_mask *
T0*
new_axis_mask 
g
disc/flatten/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
?????????
?
disc/flatten/Reshape/shapePackdisc/flatten/strided_slicedisc/flatten/Reshape/shape/1*
_output_shapes
:*

axis *
T0*
N
?
disc/flatten/ReshapeReshapedisc/MaxPool_1disc/flatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:??????????
?
4disc/dense/kernel/Initializer/truncated_normal/shapeConst*$
_class
loc:@disc/dense/kernel*
dtype0*
valueB"   ?  *
_output_shapes
:
?
3disc/dense/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *$
_class
loc:@disc/dense/kernel*
dtype0
?
5disc/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *?$=*
_output_shapes
: *
dtype0*$
_class
loc:@disc/dense/kernel
?
>disc/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal4disc/dense/kernel/Initializer/truncated_normal/shape*
T0*
seed2 *

seed *$
_class
loc:@disc/dense/kernel* 
_output_shapes
:
??*
dtype0
?
2disc/dense/kernel/Initializer/truncated_normal/mulMul>disc/dense/kernel/Initializer/truncated_normal/TruncatedNormal5disc/dense/kernel/Initializer/truncated_normal/stddev*$
_class
loc:@disc/dense/kernel*
T0* 
_output_shapes
:
??
?
.disc/dense/kernel/Initializer/truncated_normalAdd2disc/dense/kernel/Initializer/truncated_normal/mul3disc/dense/kernel/Initializer/truncated_normal/mean*
T0*$
_class
loc:@disc/dense/kernel* 
_output_shapes
:
??
?
disc/dense/kernel
VariableV2*
	container *
shape:
??*
dtype0*
shared_name *$
_class
loc:@disc/dense/kernel* 
_output_shapes
:
??
?
disc/dense/kernel/AssignAssigndisc/dense/kernel.disc/dense/kernel/Initializer/truncated_normal* 
_output_shapes
:
??*
use_locking(*
validate_shape(*
T0*$
_class
loc:@disc/dense/kernel
?
disc/dense/kernel/readIdentitydisc/dense/kernel*$
_class
loc:@disc/dense/kernel* 
_output_shapes
:
??*
T0
w
2disc/dense/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o?:*
_output_shapes
: *
dtype0
v
3disc/dense/kernel/Regularizer/l2_regularizer/L2LossL2Lossdisc/dense/kernel/read*
T0*
_output_shapes
: 
?
,disc/dense/kernel/Regularizer/l2_regularizerMul2disc/dense/kernel/Regularizer/l2_regularizer/scale3disc/dense/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
!disc/dense/bias/Initializer/zerosConst*
_output_shapes	
:?*"
_class
loc:@disc/dense/bias*
dtype0*
valueB?*    
?
disc/dense/bias
VariableV2*
dtype0*
shared_name *"
_class
loc:@disc/dense/bias*
	container *
_output_shapes	
:?*
shape:?
?
disc/dense/bias/AssignAssigndisc/dense/bias!disc/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*"
_class
loc:@disc/dense/bias*
T0
{
disc/dense/bias/readIdentitydisc/dense/bias*
T0*
_output_shapes	
:?*"
_class
loc:@disc/dense/bias
?
disc/dense/MatMulMatMuldisc/flatten/Reshapedisc/dense/kernel/read*
T0*(
_output_shapes
:??????????*
transpose_a( *
transpose_b( 
?
disc/dense/BiasAddBiasAdddisc/dense/MatMuldisc/dense/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:??????????
^
disc/dense/ReluReludisc/dense/BiasAdd*(
_output_shapes
:??????????*
T0
?
6disc/dense_1/kernel/Initializer/truncated_normal/shapeConst*&
_class
loc:@disc/dense_1/kernel*
valueB"?     *
_output_shapes
:*
dtype0
?
5disc/dense_1/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    *&
_class
loc:@disc/dense_1/kernel
?
7disc/dense_1/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *?P=*
dtype0*&
_class
loc:@disc/dense_1/kernel*
_output_shapes
: 
?
@disc/dense_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal6disc/dense_1/kernel/Initializer/truncated_normal/shape*
_output_shapes
:	?*&
_class
loc:@disc/dense_1/kernel*
seed2 *
T0*
dtype0*

seed 
?
4disc/dense_1/kernel/Initializer/truncated_normal/mulMul@disc/dense_1/kernel/Initializer/truncated_normal/TruncatedNormal7disc/dense_1/kernel/Initializer/truncated_normal/stddev*
_output_shapes
:	?*&
_class
loc:@disc/dense_1/kernel*
T0
?
0disc/dense_1/kernel/Initializer/truncated_normalAdd4disc/dense_1/kernel/Initializer/truncated_normal/mul5disc/dense_1/kernel/Initializer/truncated_normal/mean*&
_class
loc:@disc/dense_1/kernel*
T0*
_output_shapes
:	?
?
disc/dense_1/kernel
VariableV2*
dtype0*
shared_name *
	container *
shape:	?*
_output_shapes
:	?*&
_class
loc:@disc/dense_1/kernel
?
disc/dense_1/kernel/AssignAssigndisc/dense_1/kernel0disc/dense_1/kernel/Initializer/truncated_normal*&
_class
loc:@disc/dense_1/kernel*
T0*
validate_shape(*
_output_shapes
:	?*
use_locking(
?
disc/dense_1/kernel/readIdentitydisc/dense_1/kernel*
T0*
_output_shapes
:	?*&
_class
loc:@disc/dense_1/kernel
y
4disc/dense_1/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *o?:
z
5disc/dense_1/kernel/Regularizer/l2_regularizer/L2LossL2Lossdisc/dense_1/kernel/read*
_output_shapes
: *
T0
?
.disc/dense_1/kernel/Regularizer/l2_regularizerMul4disc/dense_1/kernel/Regularizer/l2_regularizer/scale5disc/dense_1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
#disc/dense_1/bias/Initializer/zerosConst*$
_class
loc:@disc/dense_1/bias*
valueB*    *
dtype0*
_output_shapes
:
?
disc/dense_1/bias
VariableV2*
_output_shapes
:*
dtype0*
shared_name *
shape:*$
_class
loc:@disc/dense_1/bias*
	container 
?
disc/dense_1/bias/AssignAssigndisc/dense_1/bias#disc/dense_1/bias/Initializer/zeros*
_output_shapes
:*
T0*
validate_shape(*$
_class
loc:@disc/dense_1/bias*
use_locking(
?
disc/dense_1/bias/readIdentitydisc/dense_1/bias*
T0*$
_class
loc:@disc/dense_1/bias*
_output_shapes
:
?
disc/dense_1/MatMulMatMuldisc/dense/Reludisc/dense_1/kernel/read*'
_output_shapes
:?????????*
T0*
transpose_a( *
transpose_b( 
?
disc/dense_1/BiasAddBiasAdddisc/dense_1/MatMuldisc/dense_1/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:?????????
\
	Sigmoid_4Sigmoiddisc/dense_1/BiasAdd*
T0*'
_output_shapes
:?????????
m
disc_1/Reshape/shapeConst*%
valueB"????         *
_output_shapes
:*
dtype0
?
disc_1/ReshapeReshapedecoder_1/generated_imagesdisc_1/Reshape/shape*
T0*/
_output_shapes
:?????????*
Tshape0
l
disc_1/conv2d/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
?
disc_1/conv2d/Conv2DConv2Ddisc_1/Reshapedisc/conv2d/kernel/read*
explicit_paddings
 */
_output_shapes
:?????????*
use_cudnn_on_gpu(*
strides
*
T0*
paddingVALID*
data_formatNHWC*
	dilations

?
disc_1/conv2d/BiasAddBiasAdddisc_1/conv2d/Conv2Ddisc/conv2d/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:?????????
k
disc_1/conv2d/ReluReludisc_1/conv2d/BiasAdd*
T0*/
_output_shapes
:?????????
?
disc_1/MaxPoolMaxPooldisc_1/conv2d/Relu*
strides
*
ksize
*
data_formatNHWC*
T0*/
_output_shapes
:?????????*
paddingSAME
n
disc_1/conv2d_1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
disc_1/conv2d_1/Conv2DConv2Ddisc_1/MaxPooldisc/conv2d_1/kernel/read*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*
explicit_paddings
 *
T0*/
_output_shapes
:?????????2*
	dilations
*
strides

?
disc_1/conv2d_1/BiasAddBiasAdddisc_1/conv2d_1/Conv2Ddisc/conv2d_1/bias/read*
data_formatNHWC*/
_output_shapes
:?????????2*
T0
o
disc_1/conv2d_1/ReluReludisc_1/conv2d_1/BiasAdd*/
_output_shapes
:?????????2*
T0
?
disc_1/MaxPool_1MaxPooldisc_1/conv2d_1/Relu*
data_formatNHWC*/
_output_shapes
:?????????2*
T0*
strides
*
paddingSAME*
ksize

d
disc_1/flatten/ShapeShapedisc_1/MaxPool_1*
out_type0*
T0*
_output_shapes
:
l
"disc_1/flatten/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
n
$disc_1/flatten/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
n
$disc_1/flatten/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
disc_1/flatten/strided_sliceStridedSlicedisc_1/flatten/Shape"disc_1/flatten/strided_slice/stack$disc_1/flatten/strided_slice/stack_1$disc_1/flatten/strided_slice/stack_2*

begin_mask *
Index0*
T0*
end_mask *
_output_shapes
: *
new_axis_mask *
shrink_axis_mask*
ellipsis_mask 
i
disc_1/flatten/Reshape/shape/1Const*
dtype0*
valueB :
?????????*
_output_shapes
: 
?
disc_1/flatten/Reshape/shapePackdisc_1/flatten/strided_slicedisc_1/flatten/Reshape/shape/1*
_output_shapes
:*

axis *
T0*
N
?
disc_1/flatten/ReshapeReshapedisc_1/MaxPool_1disc_1/flatten/Reshape/shape*
Tshape0*(
_output_shapes
:??????????*
T0
?
disc_1/dense/MatMulMatMuldisc_1/flatten/Reshapedisc/dense/kernel/read*(
_output_shapes
:??????????*
transpose_b( *
transpose_a( *
T0
?
disc_1/dense/BiasAddBiasAdddisc_1/dense/MatMuldisc/dense/bias/read*
data_formatNHWC*(
_output_shapes
:??????????*
T0
b
disc_1/dense/ReluReludisc_1/dense/BiasAdd*(
_output_shapes
:??????????*
T0
?
disc_1/dense_1/MatMulMatMuldisc_1/dense/Reludisc/dense_1/kernel/read*'
_output_shapes
:?????????*
transpose_b( *
transpose_a( *
T0
?
disc_1/dense_1/BiasAddBiasAdddisc_1/dense_1/MatMuldisc/dense_1/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:?????????
^
	Sigmoid_5Sigmoiddisc_1/dense_1/BiasAdd*
T0*'
_output_shapes
:?????????
R
ExpExpdisc/dense_1/BiasAdd*
T0*'
_output_shapes
:?????????
O
evidence_outIdentityExp*'
_output_shapes
:?????????*
T0
M
add_13/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
N
add_13AddExpadd_13/y*
T0*'
_output_shapes
:?????????
Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
|
Sum_1Sumadd_13Sum_1/reduction_indices*
	keep_dims(*'
_output_shapes
:?????????*
T0*

Tidx0
N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@
V
truedivRealDiv	truediv/xSum_1*
T0*'
_output_shapes
:?????????
V
uncertainty_outIdentitytruediv*
T0*'
_output_shapes
:?????????
Y
Sum_2/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
|
Sum_2Sumadd_13Sum_2/reduction_indices*

Tidx0*
T0*'
_output_shapes
:?????????*
	keep_dims(
U
	truediv_1RealDivadd_13Sum_2*'
_output_shapes
:?????????*
T0
Q
prob_outIdentity	truediv_1*
T0*'
_output_shapes
:?????????
?
total_regularization_lossAddN+diz/dense/kernel/Regularizer/l2_regularizer)diz/dense/bias/Regularizer/l2_regularizer-diz/dense_1/kernel/Regularizer/l2_regularizer+diz/dense_1/bias/Regularizer/l2_regularizer-diz/dense_2/kernel/Regularizer/l2_regularizer+diz/dense_2/bias/Regularizer/l2_regularizer-diz/dense_3/kernel/Regularizer/l2_regularizer+diz/dense_3/bias/Regularizer/l2_regularizer.disc0/conv2d/kernel/Regularizer/l2_regularizer0disc0/conv2d_1/kernel/Regularizer/l2_regularizer-disc0/dense/kernel/Regularizer/l2_regularizer/disc0/dense_1/kernel/Regularizer/l2_regularizer-disc/conv2d/kernel/Regularizer/l2_regularizer/disc/conv2d_1/kernel/Regularizer/l2_regularizer,disc/dense/kernel/Regularizer/l2_regularizer.disc/dense_1/kernel/Regularizer/l2_regularizer*
T0*
_output_shapes
: *
N
A
Neg_8NegY*
T0*'
_output_shapes
:?????????
M
add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+2
T
add_14Add	Sigmoid_4add_14/y*
T0*'
_output_shapes
:?????????
F
Log_6Logadd_14*
T0*'
_output_shapes
:?????????
L
mul_1MulNeg_8Log_6*'
_output_shapes
:?????????*
T0
Y
Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
w
Sum_3Summul_1Sum_3/reduction_indices*#
_output_shapes
:?????????*

Tidx0*
	keep_dims( *
T0
L
sub_4/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0
J
sub_4Subsub_4/xY*'
_output_shapes
:?????????*
T0
E
Neg_9Negsub_4*
T0*'
_output_shapes
:?????????
L
sub_5/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
R
sub_5Subsub_5/x	Sigmoid_4*'
_output_shapes
:?????????*
T0
M
add_15/yConst*
valueB
 *w?+2*
_output_shapes
: *
dtype0
P
add_15Addsub_5add_15/y*'
_output_shapes
:?????????*
T0
F
Log_7Logadd_15*'
_output_shapes
:?????????*
T0
L
mul_2MulNeg_9Log_7*'
_output_shapes
:?????????*
T0
Y
Sum_4/reduction_indicesConst*
value	B :*
_output_shapes
: *
dtype0
w
Sum_4Summul_2Sum_4/reduction_indices*#
_output_shapes
:?????????*

Tidx0*
T0*
	keep_dims( 
I
add_16AddSum_3Sum_4*#
_output_shapes
:?????????*
T0
L
sub_6/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
R
sub_6Subsub_6/x	Sigmoid_5*'
_output_shapes
:?????????*
T0
M
add_17/yConst*
valueB
 *w?+2*
dtype0*
_output_shapes
: 
P
add_17Addsub_6add_17/y*
T0*'
_output_shapes
:?????????
F
Log_8Logadd_17*
T0*'
_output_shapes
:?????????
F
Neg_10NegLog_8*
T0*'
_output_shapes
:?????????
Y
Sum_5/reduction_indicesConst*
value	B :*
_output_shapes
: *
dtype0
x
Sum_5SumNeg_10Sum_5/reduction_indices*
T0*
	keep_dims( *#
_output_shapes
:?????????*

Tidx0
J
add_18Addadd_16Sum_5*
T0*#
_output_shapes
:?????????
R
Const_10Const*
_output_shapes
:*
dtype0*
valueB: 
_
Mean_10Meanadd_18Const_10*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
B
Neg_11NegY*'
_output_shapes
:?????????*
T0
M
add_19/yConst*
_output_shapes
: *
valueB
 *??'7*
dtype0
T
add_19Add	Sigmoid_4add_19/y*'
_output_shapes
:?????????*
T0
F
Log_9Logadd_19*'
_output_shapes
:?????????*
T0
M
mul_3MulNeg_11Log_9*
T0*'
_output_shapes
:?????????
Y
Sum_6/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
w
Sum_6Summul_3Sum_6/reduction_indices*
	keep_dims( *#
_output_shapes
:?????????*
T0*

Tidx0
B
Neg_12NegY*'
_output_shapes
:?????????*
T0
L
sub_7/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0
R
sub_7Subsub_7/x	Sigmoid_5*
T0*'
_output_shapes
:?????????
M
add_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7
P
add_20Addsub_7add_20/y*
T0*'
_output_shapes
:?????????
G
Log_10Logadd_20*
T0*'
_output_shapes
:?????????
N
mul_4MulNeg_12Log_10*'
_output_shapes
:?????????*
T0
Y
Sum_7/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0
w
Sum_7Summul_4Sum_7/reduction_indices*
T0*#
_output_shapes
:?????????*

Tidx0*
	keep_dims( 
I
add_21AddSum_6Sum_7*#
_output_shapes
:?????????*
T0
R
Const_11Const*
_output_shapes
:*
valueB: *
dtype0
_
Mean_11Meanadd_21Const_11*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
M
add_22/yConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
N
add_22AddExpadd_22/y*'
_output_shapes
:?????????*
T0
M
Const_12Const*
valueB
 *    *
_output_shapes
: *
dtype0
M
EqualEqualYConst_12*'
_output_shapes
:?????????*
T0
G
WhereWhereEqual*'
_output_shapes
:?????????*
T0

g
GatherNdGatherNdadd_22Where*
Tparams0*#
_output_shapes
:?????????*
Tindices0	
^
Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"????   
k
ReshapeReshapeGatherNdReshape/shape*
T0*'
_output_shapes
:?????????*
Tshape0
e
Const_13Const*%
valueB"  ??  ??  ??*
dtype0*
_output_shapes

:
Y
Sum_8/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
}
Sum_8SumReshapeSum_8/reduction_indices*
T0*

Tidx0*'
_output_shapes
:?????????*
	keep_dims(
Y
Sum_9/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
u
Sum_9SumConst_13Sum_9/reduction_indices*
T0*
	keep_dims(*
_output_shapes

:*

Tidx0
I
LgammaLgammaSum_8*
T0*'
_output_shapes
:?????????
M
Lgamma_1LgammaReshape*'
_output_shapes
:?????????*
T0
Z
Sum_10/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
Sum_10SumLgamma_1Sum_10/reduction_indices*'
_output_shapes
:?????????*

Tidx0*
	keep_dims(*
T0
N
sub_8SubLgammaSum_10*'
_output_shapes
:?????????*
T0
E
Lgamma_2LgammaConst_13*
T0*
_output_shapes

:
Z
Sum_11/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
w
Sum_11SumLgamma_2Sum_11/reduction_indices*
	keep_dims(*
T0*

Tidx0*
_output_shapes

:
B
Lgamma_3LgammaSum_9*
T0*
_output_shapes

:
G
sub_9SubSum_11Lgamma_3*
_output_shapes

:*
T0
K
DigammaDigammaSum_8*'
_output_shapes
:?????????*
T0
O
	Digamma_1DigammaReshape*
T0*'
_output_shapes
:?????????
R
sub_10SubReshapeConst_13*'
_output_shapes
:?????????*
T0
S
sub_11Sub	Digamma_1Digamma*
T0*'
_output_shapes
:?????????
N
mul_5Mulsub_10sub_11*'
_output_shapes
:?????????*
T0
Z
Sum_12/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
}
Sum_12Summul_5Sum_12/reduction_indices*
T0*
	keep_dims(*

Tidx0*'
_output_shapes
:?????????
N
add_23AddSum_12sub_8*'
_output_shapes
:?????????*
T0
N
add_24Addadd_23sub_9*'
_output_shapes
:?????????*
T0
Y
Const_14Const*
dtype0*
_output_shapes
:*
valueB"       
_
Mean_12Meanadd_24Const_14*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
@
add_25AddMean_11Mean_12*
_output_shapes
: *
T0
Q
add_26Addadd_25total_regularization_loss*
T0*
_output_shapes
: 
T
gradients_4/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_4/grad_ys_0Const*
valueB
 *  ??*
_output_shapes
: *
dtype0
u
gradients_4/FillFillgradients_4/Shapegradients_4/grad_ys_0*

index_type0*
T0*
_output_shapes
: 
C
(gradients_4/add_26_grad/tuple/group_depsNoOp^gradients_4/Fill
?
0gradients_4/add_26_grad/tuple/control_dependencyIdentitygradients_4/Fill)^gradients_4/add_26_grad/tuple/group_deps*#
_class
loc:@gradients_4/Fill*
T0*
_output_shapes
: 
?
2gradients_4/add_26_grad/tuple/control_dependency_1Identitygradients_4/Fill)^gradients_4/add_26_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_4/Fill*
_output_shapes
: 
c
(gradients_4/add_25_grad/tuple/group_depsNoOp1^gradients_4/add_26_grad/tuple/control_dependency
?
0gradients_4/add_25_grad/tuple/control_dependencyIdentity0gradients_4/add_26_grad/tuple/control_dependency)^gradients_4/add_25_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_4/Fill*
_output_shapes
: 
?
2gradients_4/add_25_grad/tuple/control_dependency_1Identity0gradients_4/add_26_grad/tuple/control_dependency)^gradients_4/add_25_grad/tuple/group_deps*
T0*
_output_shapes
: *#
_class
loc:@gradients_4/Fill
x
;gradients_4/total_regularization_loss_grad/tuple/group_depsNoOp3^gradients_4/add_26_grad/tuple/control_dependency_1
?
Cgradients_4/total_regularization_loss_grad/tuple/control_dependencyIdentity2gradients_4/add_26_grad/tuple/control_dependency_1<^gradients_4/total_regularization_loss_grad/tuple/group_deps*#
_class
loc:@gradients_4/Fill*
T0*
_output_shapes
: 
?
Egradients_4/total_regularization_loss_grad/tuple/control_dependency_1Identity2gradients_4/add_26_grad/tuple/control_dependency_1<^gradients_4/total_regularization_loss_grad/tuple/group_deps*
_output_shapes
: *#
_class
loc:@gradients_4/Fill*
T0
?
Egradients_4/total_regularization_loss_grad/tuple/control_dependency_2Identity2gradients_4/add_26_grad/tuple/control_dependency_1<^gradients_4/total_regularization_loss_grad/tuple/group_deps*#
_class
loc:@gradients_4/Fill*
T0*
_output_shapes
: 
?
Egradients_4/total_regularization_loss_grad/tuple/control_dependency_3Identity2gradients_4/add_26_grad/tuple/control_dependency_1<^gradients_4/total_regularization_loss_grad/tuple/group_deps*#
_class
loc:@gradients_4/Fill*
T0*
_output_shapes
: 
?
Egradients_4/total_regularization_loss_grad/tuple/control_dependency_4Identity2gradients_4/add_26_grad/tuple/control_dependency_1<^gradients_4/total_regularization_loss_grad/tuple/group_deps*
_output_shapes
: *
T0*#
_class
loc:@gradients_4/Fill
?
Egradients_4/total_regularization_loss_grad/tuple/control_dependency_5Identity2gradients_4/add_26_grad/tuple/control_dependency_1<^gradients_4/total_regularization_loss_grad/tuple/group_deps*#
_class
loc:@gradients_4/Fill*
T0*
_output_shapes
: 
?
Egradients_4/total_regularization_loss_grad/tuple/control_dependency_6Identity2gradients_4/add_26_grad/tuple/control_dependency_1<^gradients_4/total_regularization_loss_grad/tuple/group_deps*#
_class
loc:@gradients_4/Fill*
_output_shapes
: *
T0
?
Egradients_4/total_regularization_loss_grad/tuple/control_dependency_7Identity2gradients_4/add_26_grad/tuple/control_dependency_1<^gradients_4/total_regularization_loss_grad/tuple/group_deps*#
_class
loc:@gradients_4/Fill*
T0*
_output_shapes
: 
?
Egradients_4/total_regularization_loss_grad/tuple/control_dependency_8Identity2gradients_4/add_26_grad/tuple/control_dependency_1<^gradients_4/total_regularization_loss_grad/tuple/group_deps*
T0*
_output_shapes
: *#
_class
loc:@gradients_4/Fill
?
Egradients_4/total_regularization_loss_grad/tuple/control_dependency_9Identity2gradients_4/add_26_grad/tuple/control_dependency_1<^gradients_4/total_regularization_loss_grad/tuple/group_deps*#
_class
loc:@gradients_4/Fill*
T0*
_output_shapes
: 
?
Fgradients_4/total_regularization_loss_grad/tuple/control_dependency_10Identity2gradients_4/add_26_grad/tuple/control_dependency_1<^gradients_4/total_regularization_loss_grad/tuple/group_deps*#
_class
loc:@gradients_4/Fill*
T0*
_output_shapes
: 
?
Fgradients_4/total_regularization_loss_grad/tuple/control_dependency_11Identity2gradients_4/add_26_grad/tuple/control_dependency_1<^gradients_4/total_regularization_loss_grad/tuple/group_deps*#
_class
loc:@gradients_4/Fill*
_output_shapes
: *
T0
?
Fgradients_4/total_regularization_loss_grad/tuple/control_dependency_12Identity2gradients_4/add_26_grad/tuple/control_dependency_1<^gradients_4/total_regularization_loss_grad/tuple/group_deps*#
_class
loc:@gradients_4/Fill*
T0*
_output_shapes
: 
?
Fgradients_4/total_regularization_loss_grad/tuple/control_dependency_13Identity2gradients_4/add_26_grad/tuple/control_dependency_1<^gradients_4/total_regularization_loss_grad/tuple/group_deps*
T0*
_output_shapes
: *#
_class
loc:@gradients_4/Fill
?
Fgradients_4/total_regularization_loss_grad/tuple/control_dependency_14Identity2gradients_4/add_26_grad/tuple/control_dependency_1<^gradients_4/total_regularization_loss_grad/tuple/group_deps*#
_class
loc:@gradients_4/Fill*
_output_shapes
: *
T0
?
Fgradients_4/total_regularization_loss_grad/tuple/control_dependency_15Identity2gradients_4/add_26_grad/tuple/control_dependency_1<^gradients_4/total_regularization_loss_grad/tuple/group_deps*
_output_shapes
: *#
_class
loc:@gradients_4/Fill*
T0
p
&gradients_4/Mean_11_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
?
 gradients_4/Mean_11_grad/ReshapeReshape0gradients_4/add_25_grad/tuple/control_dependency&gradients_4/Mean_11_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
d
gradients_4/Mean_11_grad/ShapeShapeadd_21*
_output_shapes
:*
out_type0*
T0
?
gradients_4/Mean_11_grad/TileTile gradients_4/Mean_11_grad/Reshapegradients_4/Mean_11_grad/Shape*#
_output_shapes
:?????????*

Tmultiples0*
T0
f
 gradients_4/Mean_11_grad/Shape_1Shapeadd_21*
T0*
_output_shapes
:*
out_type0
c
 gradients_4/Mean_11_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
h
gradients_4/Mean_11_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
gradients_4/Mean_11_grad/ProdProd gradients_4/Mean_11_grad/Shape_1gradients_4/Mean_11_grad/Const*

Tidx0*
_output_shapes
: *
	keep_dims( *
T0
j
 gradients_4/Mean_11_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
?
gradients_4/Mean_11_grad/Prod_1Prod gradients_4/Mean_11_grad/Shape_2 gradients_4/Mean_11_grad/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
d
"gradients_4/Mean_11_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
 gradients_4/Mean_11_grad/MaximumMaximumgradients_4/Mean_11_grad/Prod_1"gradients_4/Mean_11_grad/Maximum/y*
_output_shapes
: *
T0
?
!gradients_4/Mean_11_grad/floordivFloorDivgradients_4/Mean_11_grad/Prod gradients_4/Mean_11_grad/Maximum*
T0*
_output_shapes
: 
?
gradients_4/Mean_11_grad/CastCast!gradients_4/Mean_11_grad/floordiv*
_output_shapes
: *

SrcT0*

DstT0*
Truncate( 
?
 gradients_4/Mean_11_grad/truedivRealDivgradients_4/Mean_11_grad/Tilegradients_4/Mean_11_grad/Cast*
T0*#
_output_shapes
:?????????
w
&gradients_4/Mean_12_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
?
 gradients_4/Mean_12_grad/ReshapeReshape2gradients_4/add_25_grad/tuple/control_dependency_1&gradients_4/Mean_12_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0
d
gradients_4/Mean_12_grad/ShapeShapeadd_24*
out_type0*
_output_shapes
:*
T0
?
gradients_4/Mean_12_grad/TileTile gradients_4/Mean_12_grad/Reshapegradients_4/Mean_12_grad/Shape*

Tmultiples0*'
_output_shapes
:?????????*
T0
f
 gradients_4/Mean_12_grad/Shape_1Shapeadd_24*
T0*
out_type0*
_output_shapes
:
c
 gradients_4/Mean_12_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
h
gradients_4/Mean_12_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
gradients_4/Mean_12_grad/ProdProd gradients_4/Mean_12_grad/Shape_1gradients_4/Mean_12_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
j
 gradients_4/Mean_12_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
?
gradients_4/Mean_12_grad/Prod_1Prod gradients_4/Mean_12_grad/Shape_2 gradients_4/Mean_12_grad/Const_1*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
d
"gradients_4/Mean_12_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
?
 gradients_4/Mean_12_grad/MaximumMaximumgradients_4/Mean_12_grad/Prod_1"gradients_4/Mean_12_grad/Maximum/y*
T0*
_output_shapes
: 
?
!gradients_4/Mean_12_grad/floordivFloorDivgradients_4/Mean_12_grad/Prod gradients_4/Mean_12_grad/Maximum*
T0*
_output_shapes
: 
?
gradients_4/Mean_12_grad/CastCast!gradients_4/Mean_12_grad/floordiv*
_output_shapes
: *

DstT0*
Truncate( *

SrcT0
?
 gradients_4/Mean_12_grad/truedivRealDivgradients_4/Mean_12_grad/Tilegradients_4/Mean_12_grad/Cast*
T0*'
_output_shapes
:?????????
?
Bgradients_4/disc/conv2d/kernel/Regularizer/l2_regularizer_grad/MulMulFgradients_4/total_regularization_loss_grad/tuple/control_dependency_124disc/conv2d/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
Dgradients_4/disc/conv2d/kernel/Regularizer/l2_regularizer_grad/Mul_1MulFgradients_4/total_regularization_loss_grad/tuple/control_dependency_123disc/conv2d/kernel/Regularizer/l2_regularizer/scale*
_output_shapes
: *
T0
?
Ogradients_4/disc/conv2d/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOpC^gradients_4/disc/conv2d/kernel/Regularizer/l2_regularizer_grad/MulE^gradients_4/disc/conv2d/kernel/Regularizer/l2_regularizer_grad/Mul_1
?
Wgradients_4/disc/conv2d/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentityBgradients_4/disc/conv2d/kernel/Regularizer/l2_regularizer_grad/MulP^gradients_4/disc/conv2d/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*
_output_shapes
: *U
_classK
IGloc:@gradients_4/disc/conv2d/kernel/Regularizer/l2_regularizer_grad/Mul
?
Ygradients_4/disc/conv2d/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1IdentityDgradients_4/disc/conv2d/kernel/Regularizer/l2_regularizer_grad/Mul_1P^gradients_4/disc/conv2d/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*W
_classM
KIloc:@gradients_4/disc/conv2d/kernel/Regularizer/l2_regularizer_grad/Mul_1*
_output_shapes
: *
T0
?
Dgradients_4/disc/conv2d_1/kernel/Regularizer/l2_regularizer_grad/MulMulFgradients_4/total_regularization_loss_grad/tuple/control_dependency_136disc/conv2d_1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
Fgradients_4/disc/conv2d_1/kernel/Regularizer/l2_regularizer_grad/Mul_1MulFgradients_4/total_regularization_loss_grad/tuple/control_dependency_135disc/conv2d_1/kernel/Regularizer/l2_regularizer/scale*
_output_shapes
: *
T0
?
Qgradients_4/disc/conv2d_1/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOpE^gradients_4/disc/conv2d_1/kernel/Regularizer/l2_regularizer_grad/MulG^gradients_4/disc/conv2d_1/kernel/Regularizer/l2_regularizer_grad/Mul_1
?
Ygradients_4/disc/conv2d_1/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentityDgradients_4/disc/conv2d_1/kernel/Regularizer/l2_regularizer_grad/MulR^gradients_4/disc/conv2d_1/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_4/disc/conv2d_1/kernel/Regularizer/l2_regularizer_grad/Mul*
_output_shapes
: 
?
[gradients_4/disc/conv2d_1/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1IdentityFgradients_4/disc/conv2d_1/kernel/Regularizer/l2_regularizer_grad/Mul_1R^gradients_4/disc/conv2d_1/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*
_output_shapes
: *Y
_classO
MKloc:@gradients_4/disc/conv2d_1/kernel/Regularizer/l2_regularizer_grad/Mul_1
?
Agradients_4/disc/dense/kernel/Regularizer/l2_regularizer_grad/MulMulFgradients_4/total_regularization_loss_grad/tuple/control_dependency_143disc/dense/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
Cgradients_4/disc/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1MulFgradients_4/total_regularization_loss_grad/tuple/control_dependency_142disc/dense/kernel/Regularizer/l2_regularizer/scale*
_output_shapes
: *
T0
?
Ngradients_4/disc/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOpB^gradients_4/disc/dense/kernel/Regularizer/l2_regularizer_grad/MulD^gradients_4/disc/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1
?
Vgradients_4/disc/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentityAgradients_4/disc/dense/kernel/Regularizer/l2_regularizer_grad/MulO^gradients_4/disc/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*T
_classJ
HFloc:@gradients_4/disc/dense/kernel/Regularizer/l2_regularizer_grad/Mul*
T0*
_output_shapes
: 
?
Xgradients_4/disc/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1IdentityCgradients_4/disc/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1O^gradients_4/disc/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_4/disc/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1*
_output_shapes
: 
?
Cgradients_4/disc/dense_1/kernel/Regularizer/l2_regularizer_grad/MulMulFgradients_4/total_regularization_loss_grad/tuple/control_dependency_155disc/dense_1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
Egradients_4/disc/dense_1/kernel/Regularizer/l2_regularizer_grad/Mul_1MulFgradients_4/total_regularization_loss_grad/tuple/control_dependency_154disc/dense_1/kernel/Regularizer/l2_regularizer/scale*
_output_shapes
: *
T0
?
Pgradients_4/disc/dense_1/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOpD^gradients_4/disc/dense_1/kernel/Regularizer/l2_regularizer_grad/MulF^gradients_4/disc/dense_1/kernel/Regularizer/l2_regularizer_grad/Mul_1
?
Xgradients_4/disc/dense_1/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentityCgradients_4/disc/dense_1/kernel/Regularizer/l2_regularizer_grad/MulQ^gradients_4/disc/dense_1/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_4/disc/dense_1/kernel/Regularizer/l2_regularizer_grad/Mul*
_output_shapes
: 
?
Zgradients_4/disc/dense_1/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1IdentityEgradients_4/disc/dense_1/kernel/Regularizer/l2_regularizer_grad/Mul_1Q^gradients_4/disc/dense_1/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_4/disc/dense_1/kernel/Regularizer/l2_regularizer_grad/Mul_1*
_output_shapes
: 
b
gradients_4/add_21_grad/ShapeShapeSum_6*
out_type0*
_output_shapes
:*
T0
d
gradients_4/add_21_grad/Shape_1ShapeSum_7*
T0*
out_type0*
_output_shapes
:
?
-gradients_4/add_21_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/add_21_grad/Shapegradients_4/add_21_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_4/add_21_grad/SumSum gradients_4/Mean_11_grad/truediv-gradients_4/add_21_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
?
gradients_4/add_21_grad/ReshapeReshapegradients_4/add_21_grad/Sumgradients_4/add_21_grad/Shape*
Tshape0*
T0*#
_output_shapes
:?????????
?
gradients_4/add_21_grad/Sum_1Sum gradients_4/Mean_11_grad/truediv/gradients_4/add_21_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
?
!gradients_4/add_21_grad/Reshape_1Reshapegradients_4/add_21_grad/Sum_1gradients_4/add_21_grad/Shape_1*#
_output_shapes
:?????????*
Tshape0*
T0
v
(gradients_4/add_21_grad/tuple/group_depsNoOp ^gradients_4/add_21_grad/Reshape"^gradients_4/add_21_grad/Reshape_1
?
0gradients_4/add_21_grad/tuple/control_dependencyIdentitygradients_4/add_21_grad/Reshape)^gradients_4/add_21_grad/tuple/group_deps*2
_class(
&$loc:@gradients_4/add_21_grad/Reshape*
T0*#
_output_shapes
:?????????
?
2gradients_4/add_21_grad/tuple/control_dependency_1Identity!gradients_4/add_21_grad/Reshape_1)^gradients_4/add_21_grad/tuple/group_deps*4
_class*
(&loc:@gradients_4/add_21_grad/Reshape_1*
T0*#
_output_shapes
:?????????
c
gradients_4/add_24_grad/ShapeShapeadd_23*
_output_shapes
:*
T0*
out_type0
p
gradients_4/add_24_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
?
-gradients_4/add_24_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/add_24_grad/Shapegradients_4/add_24_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_4/add_24_grad/SumSum gradients_4/Mean_12_grad/truediv-gradients_4/add_24_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
?
gradients_4/add_24_grad/ReshapeReshapegradients_4/add_24_grad/Sumgradients_4/add_24_grad/Shape*
Tshape0*'
_output_shapes
:?????????*
T0
?
gradients_4/add_24_grad/Sum_1Sum gradients_4/Mean_12_grad/truediv/gradients_4/add_24_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
!gradients_4/add_24_grad/Reshape_1Reshapegradients_4/add_24_grad/Sum_1gradients_4/add_24_grad/Shape_1*
T0*
_output_shapes

:*
Tshape0
v
(gradients_4/add_24_grad/tuple/group_depsNoOp ^gradients_4/add_24_grad/Reshape"^gradients_4/add_24_grad/Reshape_1
?
0gradients_4/add_24_grad/tuple/control_dependencyIdentitygradients_4/add_24_grad/Reshape)^gradients_4/add_24_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_4/add_24_grad/Reshape*'
_output_shapes
:?????????
?
2gradients_4/add_24_grad/tuple/control_dependency_1Identity!gradients_4/add_24_grad/Reshape_1)^gradients_4/add_24_grad/tuple/group_deps*
_output_shapes

:*4
_class*
(&loc:@gradients_4/add_24_grad/Reshape_1*
T0
?
Igradients_4/disc/conv2d/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMuldisc/conv2d/kernel/readYgradients_4/disc/conv2d/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*&
_output_shapes
:*
T0
?
Kgradients_4/disc/conv2d_1/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMuldisc/conv2d_1/kernel/read[gradients_4/disc/conv2d_1/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*&
_output_shapes
:2*
T0
?
Hgradients_4/disc/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMuldisc/dense/kernel/readXgradients_4/disc/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
??
?
Jgradients_4/disc/dense_1/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMuldisc/dense_1/kernel/readZgradients_4/disc/dense_1/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*
_output_shapes
:	?*
T0
a
gradients_4/Sum_6_grad/ShapeShapemul_3*
T0*
out_type0*
_output_shapes
:
?
gradients_4/Sum_6_grad/SizeConst*
dtype0*/
_class%
#!loc:@gradients_4/Sum_6_grad/Shape*
value	B :*
_output_shapes
: 
?
gradients_4/Sum_6_grad/addAddSum_6/reduction_indicesgradients_4/Sum_6_grad/Size*
_output_shapes
: */
_class%
#!loc:@gradients_4/Sum_6_grad/Shape*
T0
?
gradients_4/Sum_6_grad/modFloorModgradients_4/Sum_6_grad/addgradients_4/Sum_6_grad/Size*
_output_shapes
: */
_class%
#!loc:@gradients_4/Sum_6_grad/Shape*
T0
?
gradients_4/Sum_6_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: */
_class%
#!loc:@gradients_4/Sum_6_grad/Shape
?
"gradients_4/Sum_6_grad/range/startConst*
_output_shapes
: *
value	B : *
dtype0*/
_class%
#!loc:@gradients_4/Sum_6_grad/Shape
?
"gradients_4/Sum_6_grad/range/deltaConst*
value	B :*
dtype0*/
_class%
#!loc:@gradients_4/Sum_6_grad/Shape*
_output_shapes
: 
?
gradients_4/Sum_6_grad/rangeRange"gradients_4/Sum_6_grad/range/startgradients_4/Sum_6_grad/Size"gradients_4/Sum_6_grad/range/delta*
_output_shapes
:*

Tidx0*/
_class%
#!loc:@gradients_4/Sum_6_grad/Shape
?
!gradients_4/Sum_6_grad/Fill/valueConst*
dtype0*
value	B :*
_output_shapes
: */
_class%
#!loc:@gradients_4/Sum_6_grad/Shape
?
gradients_4/Sum_6_grad/FillFillgradients_4/Sum_6_grad/Shape_1!gradients_4/Sum_6_grad/Fill/value*/
_class%
#!loc:@gradients_4/Sum_6_grad/Shape*

index_type0*
_output_shapes
: *
T0
?
$gradients_4/Sum_6_grad/DynamicStitchDynamicStitchgradients_4/Sum_6_grad/rangegradients_4/Sum_6_grad/modgradients_4/Sum_6_grad/Shapegradients_4/Sum_6_grad/Fill*
T0*
N*/
_class%
#!loc:@gradients_4/Sum_6_grad/Shape*
_output_shapes
:
?
 gradients_4/Sum_6_grad/Maximum/yConst*
value	B :*/
_class%
#!loc:@gradients_4/Sum_6_grad/Shape*
dtype0*
_output_shapes
: 
?
gradients_4/Sum_6_grad/MaximumMaximum$gradients_4/Sum_6_grad/DynamicStitch gradients_4/Sum_6_grad/Maximum/y*
T0*
_output_shapes
:*/
_class%
#!loc:@gradients_4/Sum_6_grad/Shape
?
gradients_4/Sum_6_grad/floordivFloorDivgradients_4/Sum_6_grad/Shapegradients_4/Sum_6_grad/Maximum*
T0*
_output_shapes
:*/
_class%
#!loc:@gradients_4/Sum_6_grad/Shape
?
gradients_4/Sum_6_grad/ReshapeReshape0gradients_4/add_21_grad/tuple/control_dependency$gradients_4/Sum_6_grad/DynamicStitch*0
_output_shapes
:??????????????????*
Tshape0*
T0
?
gradients_4/Sum_6_grad/TileTilegradients_4/Sum_6_grad/Reshapegradients_4/Sum_6_grad/floordiv*

Tmultiples0*'
_output_shapes
:?????????*
T0
a
gradients_4/Sum_7_grad/ShapeShapemul_4*
_output_shapes
:*
T0*
out_type0
?
gradients_4/Sum_7_grad/SizeConst*
_output_shapes
: */
_class%
#!loc:@gradients_4/Sum_7_grad/Shape*
value	B :*
dtype0
?
gradients_4/Sum_7_grad/addAddSum_7/reduction_indicesgradients_4/Sum_7_grad/Size*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients_4/Sum_7_grad/Shape
?
gradients_4/Sum_7_grad/modFloorModgradients_4/Sum_7_grad/addgradients_4/Sum_7_grad/Size*/
_class%
#!loc:@gradients_4/Sum_7_grad/Shape*
_output_shapes
: *
T0
?
gradients_4/Sum_7_grad/Shape_1Const*
dtype0*/
_class%
#!loc:@gradients_4/Sum_7_grad/Shape*
_output_shapes
: *
valueB 
?
"gradients_4/Sum_7_grad/range/startConst*
value	B : */
_class%
#!loc:@gradients_4/Sum_7_grad/Shape*
dtype0*
_output_shapes
: 
?
"gradients_4/Sum_7_grad/range/deltaConst*/
_class%
#!loc:@gradients_4/Sum_7_grad/Shape*
dtype0*
_output_shapes
: *
value	B :
?
gradients_4/Sum_7_grad/rangeRange"gradients_4/Sum_7_grad/range/startgradients_4/Sum_7_grad/Size"gradients_4/Sum_7_grad/range/delta*/
_class%
#!loc:@gradients_4/Sum_7_grad/Shape*
_output_shapes
:*

Tidx0
?
!gradients_4/Sum_7_grad/Fill/valueConst*
value	B :*
_output_shapes
: *
dtype0*/
_class%
#!loc:@gradients_4/Sum_7_grad/Shape
?
gradients_4/Sum_7_grad/FillFillgradients_4/Sum_7_grad/Shape_1!gradients_4/Sum_7_grad/Fill/value*/
_class%
#!loc:@gradients_4/Sum_7_grad/Shape*
_output_shapes
: *

index_type0*
T0
?
$gradients_4/Sum_7_grad/DynamicStitchDynamicStitchgradients_4/Sum_7_grad/rangegradients_4/Sum_7_grad/modgradients_4/Sum_7_grad/Shapegradients_4/Sum_7_grad/Fill*/
_class%
#!loc:@gradients_4/Sum_7_grad/Shape*
T0*
_output_shapes
:*
N
?
 gradients_4/Sum_7_grad/Maximum/yConst*
value	B :*
dtype0*/
_class%
#!loc:@gradients_4/Sum_7_grad/Shape*
_output_shapes
: 
?
gradients_4/Sum_7_grad/MaximumMaximum$gradients_4/Sum_7_grad/DynamicStitch gradients_4/Sum_7_grad/Maximum/y*
_output_shapes
:*
T0*/
_class%
#!loc:@gradients_4/Sum_7_grad/Shape
?
gradients_4/Sum_7_grad/floordivFloorDivgradients_4/Sum_7_grad/Shapegradients_4/Sum_7_grad/Maximum*/
_class%
#!loc:@gradients_4/Sum_7_grad/Shape*
T0*
_output_shapes
:
?
gradients_4/Sum_7_grad/ReshapeReshape2gradients_4/add_21_grad/tuple/control_dependency_1$gradients_4/Sum_7_grad/DynamicStitch*0
_output_shapes
:??????????????????*
Tshape0*
T0
?
gradients_4/Sum_7_grad/TileTilegradients_4/Sum_7_grad/Reshapegradients_4/Sum_7_grad/floordiv*'
_output_shapes
:?????????*

Tmultiples0*
T0
c
gradients_4/add_23_grad/ShapeShapeSum_12*
out_type0*
_output_shapes
:*
T0
d
gradients_4/add_23_grad/Shape_1Shapesub_8*
_output_shapes
:*
out_type0*
T0
?
-gradients_4/add_23_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/add_23_grad/Shapegradients_4/add_23_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_4/add_23_grad/SumSum0gradients_4/add_24_grad/tuple/control_dependency-gradients_4/add_23_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
gradients_4/add_23_grad/ReshapeReshapegradients_4/add_23_grad/Sumgradients_4/add_23_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients_4/add_23_grad/Sum_1Sum0gradients_4/add_24_grad/tuple/control_dependency/gradients_4/add_23_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:*

Tidx0
?
!gradients_4/add_23_grad/Reshape_1Reshapegradients_4/add_23_grad/Sum_1gradients_4/add_23_grad/Shape_1*
Tshape0*'
_output_shapes
:?????????*
T0
v
(gradients_4/add_23_grad/tuple/group_depsNoOp ^gradients_4/add_23_grad/Reshape"^gradients_4/add_23_grad/Reshape_1
?
0gradients_4/add_23_grad/tuple/control_dependencyIdentitygradients_4/add_23_grad/Reshape)^gradients_4/add_23_grad/tuple/group_deps*2
_class(
&$loc:@gradients_4/add_23_grad/Reshape*'
_output_shapes
:?????????*
T0
?
2gradients_4/add_23_grad/tuple/control_dependency_1Identity!gradients_4/add_23_grad/Reshape_1)^gradients_4/add_23_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_4/add_23_grad/Reshape_1*'
_output_shapes
:?????????
b
gradients_4/mul_3_grad/ShapeShapeNeg_11*
out_type0*
T0*
_output_shapes
:
c
gradients_4/mul_3_grad/Shape_1ShapeLog_9*
_output_shapes
:*
T0*
out_type0
?
,gradients_4/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/mul_3_grad/Shapegradients_4/mul_3_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
w
gradients_4/mul_3_grad/MulMulgradients_4/Sum_6_grad/TileLog_9*'
_output_shapes
:?????????*
T0
?
gradients_4/mul_3_grad/SumSumgradients_4/mul_3_grad/Mul,gradients_4/mul_3_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
gradients_4/mul_3_grad/ReshapeReshapegradients_4/mul_3_grad/Sumgradients_4/mul_3_grad/Shape*'
_output_shapes
:?????????*
Tshape0*
T0
z
gradients_4/mul_3_grad/Mul_1MulNeg_11gradients_4/Sum_6_grad/Tile*'
_output_shapes
:?????????*
T0
?
gradients_4/mul_3_grad/Sum_1Sumgradients_4/mul_3_grad/Mul_1.gradients_4/mul_3_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
 gradients_4/mul_3_grad/Reshape_1Reshapegradients_4/mul_3_grad/Sum_1gradients_4/mul_3_grad/Shape_1*'
_output_shapes
:?????????*
T0*
Tshape0
s
'gradients_4/mul_3_grad/tuple/group_depsNoOp^gradients_4/mul_3_grad/Reshape!^gradients_4/mul_3_grad/Reshape_1
?
/gradients_4/mul_3_grad/tuple/control_dependencyIdentitygradients_4/mul_3_grad/Reshape(^gradients_4/mul_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients_4/mul_3_grad/Reshape*'
_output_shapes
:?????????*
T0
?
1gradients_4/mul_3_grad/tuple/control_dependency_1Identity gradients_4/mul_3_grad/Reshape_1(^gradients_4/mul_3_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*3
_class)
'%loc:@gradients_4/mul_3_grad/Reshape_1
b
gradients_4/mul_4_grad/ShapeShapeNeg_12*
_output_shapes
:*
T0*
out_type0
d
gradients_4/mul_4_grad/Shape_1ShapeLog_10*
_output_shapes
:*
out_type0*
T0
?
,gradients_4/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/mul_4_grad/Shapegradients_4/mul_4_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
x
gradients_4/mul_4_grad/MulMulgradients_4/Sum_7_grad/TileLog_10*
T0*'
_output_shapes
:?????????
?
gradients_4/mul_4_grad/SumSumgradients_4/mul_4_grad/Mul,gradients_4/mul_4_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
?
gradients_4/mul_4_grad/ReshapeReshapegradients_4/mul_4_grad/Sumgradients_4/mul_4_grad/Shape*
Tshape0*
T0*'
_output_shapes
:?????????
z
gradients_4/mul_4_grad/Mul_1MulNeg_12gradients_4/Sum_7_grad/Tile*'
_output_shapes
:?????????*
T0
?
gradients_4/mul_4_grad/Sum_1Sumgradients_4/mul_4_grad/Mul_1.gradients_4/mul_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
?
 gradients_4/mul_4_grad/Reshape_1Reshapegradients_4/mul_4_grad/Sum_1gradients_4/mul_4_grad/Shape_1*'
_output_shapes
:?????????*
Tshape0*
T0
s
'gradients_4/mul_4_grad/tuple/group_depsNoOp^gradients_4/mul_4_grad/Reshape!^gradients_4/mul_4_grad/Reshape_1
?
/gradients_4/mul_4_grad/tuple/control_dependencyIdentitygradients_4/mul_4_grad/Reshape(^gradients_4/mul_4_grad/tuple/group_deps*'
_output_shapes
:?????????*1
_class'
%#loc:@gradients_4/mul_4_grad/Reshape*
T0
?
1gradients_4/mul_4_grad/tuple/control_dependency_1Identity gradients_4/mul_4_grad/Reshape_1(^gradients_4/mul_4_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*3
_class)
'%loc:@gradients_4/mul_4_grad/Reshape_1
b
gradients_4/Sum_12_grad/ShapeShapemul_5*
_output_shapes
:*
T0*
out_type0
?
gradients_4/Sum_12_grad/SizeConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@gradients_4/Sum_12_grad/Shape*
value	B :
?
gradients_4/Sum_12_grad/addAddSum_12/reduction_indicesgradients_4/Sum_12_grad/Size*
T0*0
_class&
$"loc:@gradients_4/Sum_12_grad/Shape*
_output_shapes
: 
?
gradients_4/Sum_12_grad/modFloorModgradients_4/Sum_12_grad/addgradients_4/Sum_12_grad/Size*
T0*0
_class&
$"loc:@gradients_4/Sum_12_grad/Shape*
_output_shapes
: 
?
gradients_4/Sum_12_grad/Shape_1Const*0
_class&
$"loc:@gradients_4/Sum_12_grad/Shape*
_output_shapes
: *
valueB *
dtype0
?
#gradients_4/Sum_12_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0*0
_class&
$"loc:@gradients_4/Sum_12_grad/Shape
?
#gradients_4/Sum_12_grad/range/deltaConst*
_output_shapes
: *
value	B :*0
_class&
$"loc:@gradients_4/Sum_12_grad/Shape*
dtype0
?
gradients_4/Sum_12_grad/rangeRange#gradients_4/Sum_12_grad/range/startgradients_4/Sum_12_grad/Size#gradients_4/Sum_12_grad/range/delta*

Tidx0*0
_class&
$"loc:@gradients_4/Sum_12_grad/Shape*
_output_shapes
:
?
"gradients_4/Sum_12_grad/Fill/valueConst*
_output_shapes
: *
value	B :*0
_class&
$"loc:@gradients_4/Sum_12_grad/Shape*
dtype0
?
gradients_4/Sum_12_grad/FillFillgradients_4/Sum_12_grad/Shape_1"gradients_4/Sum_12_grad/Fill/value*

index_type0*
_output_shapes
: *0
_class&
$"loc:@gradients_4/Sum_12_grad/Shape*
T0
?
%gradients_4/Sum_12_grad/DynamicStitchDynamicStitchgradients_4/Sum_12_grad/rangegradients_4/Sum_12_grad/modgradients_4/Sum_12_grad/Shapegradients_4/Sum_12_grad/Fill*
N*
T0*0
_class&
$"loc:@gradients_4/Sum_12_grad/Shape*
_output_shapes
:
?
!gradients_4/Sum_12_grad/Maximum/yConst*
dtype0*0
_class&
$"loc:@gradients_4/Sum_12_grad/Shape*
value	B :*
_output_shapes
: 
?
gradients_4/Sum_12_grad/MaximumMaximum%gradients_4/Sum_12_grad/DynamicStitch!gradients_4/Sum_12_grad/Maximum/y*
_output_shapes
:*
T0*0
_class&
$"loc:@gradients_4/Sum_12_grad/Shape
?
 gradients_4/Sum_12_grad/floordivFloorDivgradients_4/Sum_12_grad/Shapegradients_4/Sum_12_grad/Maximum*0
_class&
$"loc:@gradients_4/Sum_12_grad/Shape*
_output_shapes
:*
T0
?
gradients_4/Sum_12_grad/ReshapeReshape0gradients_4/add_23_grad/tuple/control_dependency%gradients_4/Sum_12_grad/DynamicStitch*0
_output_shapes
:??????????????????*
T0*
Tshape0
?
gradients_4/Sum_12_grad/TileTilegradients_4/Sum_12_grad/Reshape gradients_4/Sum_12_grad/floordiv*

Tmultiples0*'
_output_shapes
:?????????*
T0
b
gradients_4/sub_8_grad/ShapeShapeLgamma*
out_type0*
T0*
_output_shapes
:
d
gradients_4/sub_8_grad/Shape_1ShapeSum_10*
_output_shapes
:*
T0*
out_type0
?
,gradients_4/sub_8_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/sub_8_grad/Shapegradients_4/sub_8_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_4/sub_8_grad/SumSum2gradients_4/add_23_grad/tuple/control_dependency_1,gradients_4/sub_8_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
?
gradients_4/sub_8_grad/ReshapeReshapegradients_4/sub_8_grad/Sumgradients_4/sub_8_grad/Shape*'
_output_shapes
:?????????*
T0*
Tshape0
?
gradients_4/sub_8_grad/Sum_1Sum2gradients_4/add_23_grad/tuple/control_dependency_1.gradients_4/sub_8_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
b
gradients_4/sub_8_grad/NegNeggradients_4/sub_8_grad/Sum_1*
T0*
_output_shapes
:
?
 gradients_4/sub_8_grad/Reshape_1Reshapegradients_4/sub_8_grad/Neggradients_4/sub_8_grad/Shape_1*'
_output_shapes
:?????????*
T0*
Tshape0
s
'gradients_4/sub_8_grad/tuple/group_depsNoOp^gradients_4/sub_8_grad/Reshape!^gradients_4/sub_8_grad/Reshape_1
?
/gradients_4/sub_8_grad/tuple/control_dependencyIdentitygradients_4/sub_8_grad/Reshape(^gradients_4/sub_8_grad/tuple/group_deps*1
_class'
%#loc:@gradients_4/sub_8_grad/Reshape*
T0*'
_output_shapes
:?????????
?
1gradients_4/sub_8_grad/tuple/control_dependency_1Identity gradients_4/sub_8_grad/Reshape_1(^gradients_4/sub_8_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_4/sub_8_grad/Reshape_1*'
_output_shapes
:?????????
?
!gradients_4/Log_9_grad/Reciprocal
Reciprocaladd_192^gradients_4/mul_3_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
gradients_4/Log_9_grad/mulMul1gradients_4/mul_3_grad/tuple/control_dependency_1!gradients_4/Log_9_grad/Reciprocal*'
_output_shapes
:?????????*
T0
?
"gradients_4/Log_10_grad/Reciprocal
Reciprocaladd_202^gradients_4/mul_4_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
gradients_4/Log_10_grad/mulMul1gradients_4/mul_4_grad/tuple/control_dependency_1"gradients_4/Log_10_grad/Reciprocal*
T0*'
_output_shapes
:?????????
b
gradients_4/mul_5_grad/ShapeShapesub_10*
T0*
_output_shapes
:*
out_type0
d
gradients_4/mul_5_grad/Shape_1Shapesub_11*
T0*
out_type0*
_output_shapes
:
?
,gradients_4/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/mul_5_grad/Shapegradients_4/mul_5_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
y
gradients_4/mul_5_grad/MulMulgradients_4/Sum_12_grad/Tilesub_11*
T0*'
_output_shapes
:?????????
?
gradients_4/mul_5_grad/SumSumgradients_4/mul_5_grad/Mul,gradients_4/mul_5_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
gradients_4/mul_5_grad/ReshapeReshapegradients_4/mul_5_grad/Sumgradients_4/mul_5_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
{
gradients_4/mul_5_grad/Mul_1Mulsub_10gradients_4/Sum_12_grad/Tile*
T0*'
_output_shapes
:?????????
?
gradients_4/mul_5_grad/Sum_1Sumgradients_4/mul_5_grad/Mul_1.gradients_4/mul_5_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
 gradients_4/mul_5_grad/Reshape_1Reshapegradients_4/mul_5_grad/Sum_1gradients_4/mul_5_grad/Shape_1*
T0*'
_output_shapes
:?????????*
Tshape0
s
'gradients_4/mul_5_grad/tuple/group_depsNoOp^gradients_4/mul_5_grad/Reshape!^gradients_4/mul_5_grad/Reshape_1
?
/gradients_4/mul_5_grad/tuple/control_dependencyIdentitygradients_4/mul_5_grad/Reshape(^gradients_4/mul_5_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_4/mul_5_grad/Reshape*'
_output_shapes
:?????????
?
1gradients_4/mul_5_grad/tuple/control_dependency_1Identity gradients_4/mul_5_grad/Reshape_1(^gradients_4/mul_5_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_4/mul_5_grad/Reshape_1*'
_output_shapes
:?????????
?
gradients_4/Lgamma_grad/DigammaDigammaSum_80^gradients_4/sub_8_grad/tuple/control_dependency*'
_output_shapes
:?????????*
T0
?
gradients_4/Lgamma_grad/mulMul/gradients_4/sub_8_grad/tuple/control_dependencygradients_4/Lgamma_grad/Digamma*
T0*'
_output_shapes
:?????????
e
gradients_4/Sum_10_grad/ShapeShapeLgamma_1*
out_type0*
_output_shapes
:*
T0
?
gradients_4/Sum_10_grad/SizeConst*
_output_shapes
: *0
_class&
$"loc:@gradients_4/Sum_10_grad/Shape*
dtype0*
value	B :
?
gradients_4/Sum_10_grad/addAddSum_10/reduction_indicesgradients_4/Sum_10_grad/Size*0
_class&
$"loc:@gradients_4/Sum_10_grad/Shape*
_output_shapes
: *
T0
?
gradients_4/Sum_10_grad/modFloorModgradients_4/Sum_10_grad/addgradients_4/Sum_10_grad/Size*
T0*
_output_shapes
: *0
_class&
$"loc:@gradients_4/Sum_10_grad/Shape
?
gradients_4/Sum_10_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: *0
_class&
$"loc:@gradients_4/Sum_10_grad/Shape
?
#gradients_4/Sum_10_grad/range/startConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@gradients_4/Sum_10_grad/Shape*
value	B : 
?
#gradients_4/Sum_10_grad/range/deltaConst*
_output_shapes
: *
value	B :*0
_class&
$"loc:@gradients_4/Sum_10_grad/Shape*
dtype0
?
gradients_4/Sum_10_grad/rangeRange#gradients_4/Sum_10_grad/range/startgradients_4/Sum_10_grad/Size#gradients_4/Sum_10_grad/range/delta*
_output_shapes
:*

Tidx0*0
_class&
$"loc:@gradients_4/Sum_10_grad/Shape
?
"gradients_4/Sum_10_grad/Fill/valueConst*
dtype0*0
_class&
$"loc:@gradients_4/Sum_10_grad/Shape*
value	B :*
_output_shapes
: 
?
gradients_4/Sum_10_grad/FillFillgradients_4/Sum_10_grad/Shape_1"gradients_4/Sum_10_grad/Fill/value*
_output_shapes
: *

index_type0*0
_class&
$"loc:@gradients_4/Sum_10_grad/Shape*
T0
?
%gradients_4/Sum_10_grad/DynamicStitchDynamicStitchgradients_4/Sum_10_grad/rangegradients_4/Sum_10_grad/modgradients_4/Sum_10_grad/Shapegradients_4/Sum_10_grad/Fill*
N*0
_class&
$"loc:@gradients_4/Sum_10_grad/Shape*
T0*
_output_shapes
:
?
!gradients_4/Sum_10_grad/Maximum/yConst*
_output_shapes
: *0
_class&
$"loc:@gradients_4/Sum_10_grad/Shape*
dtype0*
value	B :
?
gradients_4/Sum_10_grad/MaximumMaximum%gradients_4/Sum_10_grad/DynamicStitch!gradients_4/Sum_10_grad/Maximum/y*
T0*
_output_shapes
:*0
_class&
$"loc:@gradients_4/Sum_10_grad/Shape
?
 gradients_4/Sum_10_grad/floordivFloorDivgradients_4/Sum_10_grad/Shapegradients_4/Sum_10_grad/Maximum*
T0*0
_class&
$"loc:@gradients_4/Sum_10_grad/Shape*
_output_shapes
:
?
gradients_4/Sum_10_grad/ReshapeReshape1gradients_4/sub_8_grad/tuple/control_dependency_1%gradients_4/Sum_10_grad/DynamicStitch*0
_output_shapes
:??????????????????*
Tshape0*
T0
?
gradients_4/Sum_10_grad/TileTilegradients_4/Sum_10_grad/Reshape gradients_4/Sum_10_grad/floordiv*

Tmultiples0*'
_output_shapes
:?????????*
T0
f
gradients_4/add_19_grad/ShapeShape	Sigmoid_4*
out_type0*
T0*
_output_shapes
:
b
gradients_4/add_19_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
?
-gradients_4/add_19_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/add_19_grad/Shapegradients_4/add_19_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_4/add_19_grad/SumSumgradients_4/Log_9_grad/mul-gradients_4/add_19_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
gradients_4/add_19_grad/ReshapeReshapegradients_4/add_19_grad/Sumgradients_4/add_19_grad/Shape*'
_output_shapes
:?????????*
T0*
Tshape0
?
gradients_4/add_19_grad/Sum_1Sumgradients_4/Log_9_grad/mul/gradients_4/add_19_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:*

Tidx0
?
!gradients_4/add_19_grad/Reshape_1Reshapegradients_4/add_19_grad/Sum_1gradients_4/add_19_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
v
(gradients_4/add_19_grad/tuple/group_depsNoOp ^gradients_4/add_19_grad/Reshape"^gradients_4/add_19_grad/Reshape_1
?
0gradients_4/add_19_grad/tuple/control_dependencyIdentitygradients_4/add_19_grad/Reshape)^gradients_4/add_19_grad/tuple/group_deps*2
_class(
&$loc:@gradients_4/add_19_grad/Reshape*
T0*'
_output_shapes
:?????????
?
2gradients_4/add_19_grad/tuple/control_dependency_1Identity!gradients_4/add_19_grad/Reshape_1)^gradients_4/add_19_grad/tuple/group_deps*
T0*
_output_shapes
: *4
_class*
(&loc:@gradients_4/add_19_grad/Reshape_1
b
gradients_4/add_20_grad/ShapeShapesub_7*
T0*
out_type0*
_output_shapes
:
b
gradients_4/add_20_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
?
-gradients_4/add_20_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/add_20_grad/Shapegradients_4/add_20_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_4/add_20_grad/SumSumgradients_4/Log_10_grad/mul-gradients_4/add_20_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
?
gradients_4/add_20_grad/ReshapeReshapegradients_4/add_20_grad/Sumgradients_4/add_20_grad/Shape*'
_output_shapes
:?????????*
Tshape0*
T0
?
gradients_4/add_20_grad/Sum_1Sumgradients_4/Log_10_grad/mul/gradients_4/add_20_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
?
!gradients_4/add_20_grad/Reshape_1Reshapegradients_4/add_20_grad/Sum_1gradients_4/add_20_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
v
(gradients_4/add_20_grad/tuple/group_depsNoOp ^gradients_4/add_20_grad/Reshape"^gradients_4/add_20_grad/Reshape_1
?
0gradients_4/add_20_grad/tuple/control_dependencyIdentitygradients_4/add_20_grad/Reshape)^gradients_4/add_20_grad/tuple/group_deps*'
_output_shapes
:?????????*
T0*2
_class(
&$loc:@gradients_4/add_20_grad/Reshape
?
2gradients_4/add_20_grad/tuple/control_dependency_1Identity!gradients_4/add_20_grad/Reshape_1)^gradients_4/add_20_grad/tuple/group_deps*
_output_shapes
: *4
_class*
(&loc:@gradients_4/add_20_grad/Reshape_1*
T0
d
gradients_4/sub_10_grad/ShapeShapeReshape*
_output_shapes
:*
out_type0*
T0
p
gradients_4/sub_10_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"      
?
-gradients_4/sub_10_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/sub_10_grad/Shapegradients_4/sub_10_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_4/sub_10_grad/SumSum/gradients_4/mul_5_grad/tuple/control_dependency-gradients_4/sub_10_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
?
gradients_4/sub_10_grad/ReshapeReshapegradients_4/sub_10_grad/Sumgradients_4/sub_10_grad/Shape*'
_output_shapes
:?????????*
Tshape0*
T0
?
gradients_4/sub_10_grad/Sum_1Sum/gradients_4/mul_5_grad/tuple/control_dependency/gradients_4/sub_10_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
d
gradients_4/sub_10_grad/NegNeggradients_4/sub_10_grad/Sum_1*
T0*
_output_shapes
:
?
!gradients_4/sub_10_grad/Reshape_1Reshapegradients_4/sub_10_grad/Neggradients_4/sub_10_grad/Shape_1*
Tshape0*
T0*
_output_shapes

:
v
(gradients_4/sub_10_grad/tuple/group_depsNoOp ^gradients_4/sub_10_grad/Reshape"^gradients_4/sub_10_grad/Reshape_1
?
0gradients_4/sub_10_grad/tuple/control_dependencyIdentitygradients_4/sub_10_grad/Reshape)^gradients_4/sub_10_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*2
_class(
&$loc:@gradients_4/sub_10_grad/Reshape
?
2gradients_4/sub_10_grad/tuple/control_dependency_1Identity!gradients_4/sub_10_grad/Reshape_1)^gradients_4/sub_10_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_4/sub_10_grad/Reshape_1*
_output_shapes

:
f
gradients_4/sub_11_grad/ShapeShape	Digamma_1*
_output_shapes
:*
T0*
out_type0
f
gradients_4/sub_11_grad/Shape_1ShapeDigamma*
T0*
out_type0*
_output_shapes
:
?
-gradients_4/sub_11_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/sub_11_grad/Shapegradients_4/sub_11_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_4/sub_11_grad/SumSum1gradients_4/mul_5_grad/tuple/control_dependency_1-gradients_4/sub_11_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
?
gradients_4/sub_11_grad/ReshapeReshapegradients_4/sub_11_grad/Sumgradients_4/sub_11_grad/Shape*
Tshape0*'
_output_shapes
:?????????*
T0
?
gradients_4/sub_11_grad/Sum_1Sum1gradients_4/mul_5_grad/tuple/control_dependency_1/gradients_4/sub_11_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:*

Tidx0
d
gradients_4/sub_11_grad/NegNeggradients_4/sub_11_grad/Sum_1*
_output_shapes
:*
T0
?
!gradients_4/sub_11_grad/Reshape_1Reshapegradients_4/sub_11_grad/Neggradients_4/sub_11_grad/Shape_1*'
_output_shapes
:?????????*
Tshape0*
T0
v
(gradients_4/sub_11_grad/tuple/group_depsNoOp ^gradients_4/sub_11_grad/Reshape"^gradients_4/sub_11_grad/Reshape_1
?
0gradients_4/sub_11_grad/tuple/control_dependencyIdentitygradients_4/sub_11_grad/Reshape)^gradients_4/sub_11_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_4/sub_11_grad/Reshape*'
_output_shapes
:?????????
?
2gradients_4/sub_11_grad/tuple/control_dependency_1Identity!gradients_4/sub_11_grad/Reshape_1)^gradients_4/sub_11_grad/tuple/group_deps*'
_output_shapes
:?????????*4
_class*
(&loc:@gradients_4/sub_11_grad/Reshape_1*
T0
?
!gradients_4/Lgamma_1_grad/DigammaDigammaReshape^gradients_4/Sum_10_grad/Tile*
T0*'
_output_shapes
:?????????
?
gradients_4/Lgamma_1_grad/mulMulgradients_4/Sum_10_grad/Tile!gradients_4/Lgamma_1_grad/Digamma*
T0*'
_output_shapes
:?????????
?
&gradients_4/Sigmoid_4_grad/SigmoidGradSigmoidGrad	Sigmoid_40gradients_4/add_19_grad/tuple/control_dependency*'
_output_shapes
:?????????*
T0
_
gradients_4/sub_7_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
g
gradients_4/sub_7_grad/Shape_1Shape	Sigmoid_5*
out_type0*
_output_shapes
:*
T0
?
,gradients_4/sub_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/sub_7_grad/Shapegradients_4/sub_7_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_4/sub_7_grad/SumSum0gradients_4/add_20_grad/tuple/control_dependency,gradients_4/sub_7_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
gradients_4/sub_7_grad/ReshapeReshapegradients_4/sub_7_grad/Sumgradients_4/sub_7_grad/Shape*
_output_shapes
: *
Tshape0*
T0
?
gradients_4/sub_7_grad/Sum_1Sum0gradients_4/add_20_grad/tuple/control_dependency.gradients_4/sub_7_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
b
gradients_4/sub_7_grad/NegNeggradients_4/sub_7_grad/Sum_1*
T0*
_output_shapes
:
?
 gradients_4/sub_7_grad/Reshape_1Reshapegradients_4/sub_7_grad/Neggradients_4/sub_7_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
s
'gradients_4/sub_7_grad/tuple/group_depsNoOp^gradients_4/sub_7_grad/Reshape!^gradients_4/sub_7_grad/Reshape_1
?
/gradients_4/sub_7_grad/tuple/control_dependencyIdentitygradients_4/sub_7_grad/Reshape(^gradients_4/sub_7_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_4/sub_7_grad/Reshape*
_output_shapes
: 
?
1gradients_4/sub_7_grad/tuple/control_dependency_1Identity gradients_4/sub_7_grad/Reshape_1(^gradients_4/sub_7_grad/tuple/group_deps*3
_class)
'%loc:@gradients_4/sub_7_grad/Reshape_1*'
_output_shapes
:?????????*
T0
?
 gradients_4/Digamma_1_grad/ConstConst1^gradients_4/sub_11_grad/tuple/control_dependency*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
$gradients_4/Digamma_1_grad/Polygamma	Polygamma gradients_4/Digamma_1_grad/ConstReshape*
T0*'
_output_shapes
:?????????
?
gradients_4/Digamma_1_grad/mulMul0gradients_4/sub_11_grad/tuple/control_dependency$gradients_4/Digamma_1_grad/Polygamma*'
_output_shapes
:?????????*
T0
?
gradients_4/Digamma_grad/ConstConst3^gradients_4/sub_11_grad/tuple/control_dependency_1*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
"gradients_4/Digamma_grad/Polygamma	Polygammagradients_4/Digamma_grad/ConstSum_8*'
_output_shapes
:?????????*
T0
?
gradients_4/Digamma_grad/mulMul2gradients_4/sub_11_grad/tuple/control_dependency_1"gradients_4/Digamma_grad/Polygamma*
T0*'
_output_shapes
:?????????
?
&gradients_4/Sigmoid_5_grad/SigmoidGradSigmoidGrad	Sigmoid_51gradients_4/sub_7_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
gradients_4/AddNAddNgradients_4/Lgamma_grad/mulgradients_4/Digamma_grad/mul*
N*'
_output_shapes
:?????????*
T0*.
_class$
" loc:@gradients_4/Lgamma_grad/mul
c
gradients_4/Sum_8_grad/ShapeShapeReshape*
_output_shapes
:*
out_type0*
T0
?
gradients_4/Sum_8_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*/
_class%
#!loc:@gradients_4/Sum_8_grad/Shape
?
gradients_4/Sum_8_grad/addAddSum_8/reduction_indicesgradients_4/Sum_8_grad/Size*/
_class%
#!loc:@gradients_4/Sum_8_grad/Shape*
T0*
_output_shapes
: 
?
gradients_4/Sum_8_grad/modFloorModgradients_4/Sum_8_grad/addgradients_4/Sum_8_grad/Size*/
_class%
#!loc:@gradients_4/Sum_8_grad/Shape*
T0*
_output_shapes
: 
?
gradients_4/Sum_8_grad/Shape_1Const*
_output_shapes
: */
_class%
#!loc:@gradients_4/Sum_8_grad/Shape*
valueB *
dtype0
?
"gradients_4/Sum_8_grad/range/startConst*
_output_shapes
: *
value	B : */
_class%
#!loc:@gradients_4/Sum_8_grad/Shape*
dtype0
?
"gradients_4/Sum_8_grad/range/deltaConst*/
_class%
#!loc:@gradients_4/Sum_8_grad/Shape*
dtype0*
_output_shapes
: *
value	B :
?
gradients_4/Sum_8_grad/rangeRange"gradients_4/Sum_8_grad/range/startgradients_4/Sum_8_grad/Size"gradients_4/Sum_8_grad/range/delta*
_output_shapes
:*/
_class%
#!loc:@gradients_4/Sum_8_grad/Shape*

Tidx0
?
!gradients_4/Sum_8_grad/Fill/valueConst*
dtype0*/
_class%
#!loc:@gradients_4/Sum_8_grad/Shape*
value	B :*
_output_shapes
: 
?
gradients_4/Sum_8_grad/FillFillgradients_4/Sum_8_grad/Shape_1!gradients_4/Sum_8_grad/Fill/value*/
_class%
#!loc:@gradients_4/Sum_8_grad/Shape*

index_type0*
_output_shapes
: *
T0
?
$gradients_4/Sum_8_grad/DynamicStitchDynamicStitchgradients_4/Sum_8_grad/rangegradients_4/Sum_8_grad/modgradients_4/Sum_8_grad/Shapegradients_4/Sum_8_grad/Fill*
N*
_output_shapes
:*/
_class%
#!loc:@gradients_4/Sum_8_grad/Shape*
T0
?
 gradients_4/Sum_8_grad/Maximum/yConst*
value	B :*/
_class%
#!loc:@gradients_4/Sum_8_grad/Shape*
_output_shapes
: *
dtype0
?
gradients_4/Sum_8_grad/MaximumMaximum$gradients_4/Sum_8_grad/DynamicStitch gradients_4/Sum_8_grad/Maximum/y*
_output_shapes
:*
T0*/
_class%
#!loc:@gradients_4/Sum_8_grad/Shape
?
gradients_4/Sum_8_grad/floordivFloorDivgradients_4/Sum_8_grad/Shapegradients_4/Sum_8_grad/Maximum*/
_class%
#!loc:@gradients_4/Sum_8_grad/Shape*
T0*
_output_shapes
:
?
gradients_4/Sum_8_grad/ReshapeReshapegradients_4/AddN$gradients_4/Sum_8_grad/DynamicStitch*0
_output_shapes
:??????????????????*
Tshape0*
T0
?
gradients_4/Sum_8_grad/TileTilegradients_4/Sum_8_grad/Reshapegradients_4/Sum_8_grad/floordiv*
T0*

Tmultiples0*'
_output_shapes
:?????????
?
3gradients_4/disc_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad&gradients_4/Sigmoid_5_grad/SigmoidGrad*
_output_shapes
:*
T0*
data_formatNHWC
?
8gradients_4/disc_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp'^gradients_4/Sigmoid_5_grad/SigmoidGrad4^gradients_4/disc_1/dense_1/BiasAdd_grad/BiasAddGrad
?
@gradients_4/disc_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity&gradients_4/Sigmoid_5_grad/SigmoidGrad9^gradients_4/disc_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_4/Sigmoid_5_grad/SigmoidGrad*'
_output_shapes
:?????????
?
Bgradients_4/disc_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity3gradients_4/disc_1/dense_1/BiasAdd_grad/BiasAddGrad9^gradients_4/disc_1/dense_1/BiasAdd_grad/tuple/group_deps*F
_class<
:8loc:@gradients_4/disc_1/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
?
gradients_4/AddN_1AddN0gradients_4/sub_10_grad/tuple/control_dependencygradients_4/Lgamma_1_grad/mulgradients_4/Digamma_1_grad/mulgradients_4/Sum_8_grad/Tile*
N*2
_class(
&$loc:@gradients_4/sub_10_grad/Reshape*'
_output_shapes
:?????????*
T0
f
gradients_4/Reshape_grad/ShapeShapeGatherNd*
T0*
out_type0*
_output_shapes
:
?
 gradients_4/Reshape_grad/ReshapeReshapegradients_4/AddN_1gradients_4/Reshape_grad/Shape*
Tshape0*#
_output_shapes
:?????????*
T0
?
-gradients_4/disc_1/dense_1/MatMul_grad/MatMulMatMul@gradients_4/disc_1/dense_1/BiasAdd_grad/tuple/control_dependencydisc/dense_1/kernel/read*(
_output_shapes
:??????????*
transpose_b(*
transpose_a( *
T0
?
/gradients_4/disc_1/dense_1/MatMul_grad/MatMul_1MatMuldisc_1/dense/Relu@gradients_4/disc_1/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	?*
transpose_a(
?
7gradients_4/disc_1/dense_1/MatMul_grad/tuple/group_depsNoOp.^gradients_4/disc_1/dense_1/MatMul_grad/MatMul0^gradients_4/disc_1/dense_1/MatMul_grad/MatMul_1
?
?gradients_4/disc_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity-gradients_4/disc_1/dense_1/MatMul_grad/MatMul8^gradients_4/disc_1/dense_1/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:??????????*@
_class6
42loc:@gradients_4/disc_1/dense_1/MatMul_grad/MatMul
?
Agradients_4/disc_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity/gradients_4/disc_1/dense_1/MatMul_grad/MatMul_18^gradients_4/disc_1/dense_1/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	?*B
_class8
64loc:@gradients_4/disc_1/dense_1/MatMul_grad/MatMul_1
e
gradients_4/GatherNd_grad/ShapeShapeadd_22*
T0*
out_type0	*
_output_shapes
:
?
#gradients_4/GatherNd_grad/ScatterNd	ScatterNdWhere gradients_4/Reshape_grad/Reshapegradients_4/GatherNd_grad/Shape*
Tindices0	*
T0*'
_output_shapes
:?????????
?
+gradients_4/disc_1/dense/Relu_grad/ReluGradReluGrad?gradients_4/disc_1/dense_1/MatMul_grad/tuple/control_dependencydisc_1/dense/Relu*
T0*(
_output_shapes
:??????????
`
gradients_4/add_22_grad/ShapeShapeExp*
_output_shapes
:*
out_type0*
T0
b
gradients_4/add_22_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
-gradients_4/add_22_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/add_22_grad/Shapegradients_4/add_22_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_4/add_22_grad/SumSum#gradients_4/GatherNd_grad/ScatterNd-gradients_4/add_22_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
?
gradients_4/add_22_grad/ReshapeReshapegradients_4/add_22_grad/Sumgradients_4/add_22_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients_4/add_22_grad/Sum_1Sum#gradients_4/GatherNd_grad/ScatterNd/gradients_4/add_22_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
?
!gradients_4/add_22_grad/Reshape_1Reshapegradients_4/add_22_grad/Sum_1gradients_4/add_22_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
v
(gradients_4/add_22_grad/tuple/group_depsNoOp ^gradients_4/add_22_grad/Reshape"^gradients_4/add_22_grad/Reshape_1
?
0gradients_4/add_22_grad/tuple/control_dependencyIdentitygradients_4/add_22_grad/Reshape)^gradients_4/add_22_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*2
_class(
&$loc:@gradients_4/add_22_grad/Reshape
?
2gradients_4/add_22_grad/tuple/control_dependency_1Identity!gradients_4/add_22_grad/Reshape_1)^gradients_4/add_22_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_4/add_22_grad/Reshape_1*
_output_shapes
: 
?
1gradients_4/disc_1/dense/BiasAdd_grad/BiasAddGradBiasAddGrad+gradients_4/disc_1/dense/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes	
:?
?
6gradients_4/disc_1/dense/BiasAdd_grad/tuple/group_depsNoOp2^gradients_4/disc_1/dense/BiasAdd_grad/BiasAddGrad,^gradients_4/disc_1/dense/Relu_grad/ReluGrad
?
>gradients_4/disc_1/dense/BiasAdd_grad/tuple/control_dependencyIdentity+gradients_4/disc_1/dense/Relu_grad/ReluGrad7^gradients_4/disc_1/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:??????????*>
_class4
20loc:@gradients_4/disc_1/dense/Relu_grad/ReluGrad*
T0
?
@gradients_4/disc_1/dense/BiasAdd_grad/tuple/control_dependency_1Identity1gradients_4/disc_1/dense/BiasAdd_grad/BiasAddGrad7^gradients_4/disc_1/dense/BiasAdd_grad/tuple/group_deps*D
_class:
86loc:@gradients_4/disc_1/dense/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:?
?
gradients_4/Exp_grad/mulMul0gradients_4/add_22_grad/tuple/control_dependencyExp*
T0*'
_output_shapes
:?????????
?
+gradients_4/disc_1/dense/MatMul_grad/MatMulMatMul>gradients_4/disc_1/dense/BiasAdd_grad/tuple/control_dependencydisc/dense/kernel/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:??????????
?
-gradients_4/disc_1/dense/MatMul_grad/MatMul_1MatMuldisc_1/flatten/Reshape>gradients_4/disc_1/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:
??
?
5gradients_4/disc_1/dense/MatMul_grad/tuple/group_depsNoOp,^gradients_4/disc_1/dense/MatMul_grad/MatMul.^gradients_4/disc_1/dense/MatMul_grad/MatMul_1
?
=gradients_4/disc_1/dense/MatMul_grad/tuple/control_dependencyIdentity+gradients_4/disc_1/dense/MatMul_grad/MatMul6^gradients_4/disc_1/dense/MatMul_grad/tuple/group_deps*(
_output_shapes
:??????????*
T0*>
_class4
20loc:@gradients_4/disc_1/dense/MatMul_grad/MatMul
?
?gradients_4/disc_1/dense/MatMul_grad/tuple/control_dependency_1Identity-gradients_4/disc_1/dense/MatMul_grad/MatMul_16^gradients_4/disc_1/dense/MatMul_grad/tuple/group_deps*@
_class6
42loc:@gradients_4/disc_1/dense/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
??
?
gradients_4/AddN_2AddN&gradients_4/Sigmoid_4_grad/SigmoidGradgradients_4/Exp_grad/mul*
T0*
N*9
_class/
-+loc:@gradients_4/Sigmoid_4_grad/SigmoidGrad*'
_output_shapes
:?????????
?
1gradients_4/disc/dense_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_4/AddN_2*
_output_shapes
:*
data_formatNHWC*
T0
?
6gradients_4/disc/dense_1/BiasAdd_grad/tuple/group_depsNoOp^gradients_4/AddN_22^gradients_4/disc/dense_1/BiasAdd_grad/BiasAddGrad
?
>gradients_4/disc/dense_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients_4/AddN_27^gradients_4/disc/dense_1/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*9
_class/
-+loc:@gradients_4/Sigmoid_4_grad/SigmoidGrad
?
@gradients_4/disc/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity1gradients_4/disc/dense_1/BiasAdd_grad/BiasAddGrad7^gradients_4/disc/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*D
_class:
86loc:@gradients_4/disc/dense_1/BiasAdd_grad/BiasAddGrad*
T0
}
-gradients_4/disc_1/flatten/Reshape_grad/ShapeShapedisc_1/MaxPool_1*
_output_shapes
:*
out_type0*
T0
?
/gradients_4/disc_1/flatten/Reshape_grad/ReshapeReshape=gradients_4/disc_1/dense/MatMul_grad/tuple/control_dependency-gradients_4/disc_1/flatten/Reshape_grad/Shape*
Tshape0*/
_output_shapes
:?????????2*
T0
?
+gradients_4/disc/dense_1/MatMul_grad/MatMulMatMul>gradients_4/disc/dense_1/BiasAdd_grad/tuple/control_dependencydisc/dense_1/kernel/read*
transpose_a( *
T0*(
_output_shapes
:??????????*
transpose_b(
?
-gradients_4/disc/dense_1/MatMul_grad/MatMul_1MatMuldisc/dense/Relu>gradients_4/disc/dense_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	?*
transpose_b( *
transpose_a(*
T0
?
5gradients_4/disc/dense_1/MatMul_grad/tuple/group_depsNoOp,^gradients_4/disc/dense_1/MatMul_grad/MatMul.^gradients_4/disc/dense_1/MatMul_grad/MatMul_1
?
=gradients_4/disc/dense_1/MatMul_grad/tuple/control_dependencyIdentity+gradients_4/disc/dense_1/MatMul_grad/MatMul6^gradients_4/disc/dense_1/MatMul_grad/tuple/group_deps*>
_class4
20loc:@gradients_4/disc/dense_1/MatMul_grad/MatMul*(
_output_shapes
:??????????*
T0
?
?gradients_4/disc/dense_1/MatMul_grad/tuple/control_dependency_1Identity-gradients_4/disc/dense_1/MatMul_grad/MatMul_16^gradients_4/disc/dense_1/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_4/disc/dense_1/MatMul_grad/MatMul_1*
_output_shapes
:	?
?
gradients_4/AddN_3AddNBgradients_4/disc_1/dense_1/BiasAdd_grad/tuple/control_dependency_1@gradients_4/disc/dense_1/BiasAdd_grad/tuple/control_dependency_1*
N*F
_class<
:8loc:@gradients_4/disc_1/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
?
-gradients_4/disc_1/MaxPool_1_grad/MaxPoolGradMaxPoolGraddisc_1/conv2d_1/Reludisc_1/MaxPool_1/gradients_4/disc_1/flatten/Reshape_grad/Reshape*
ksize
*
paddingSAME*/
_output_shapes
:?????????2*
data_formatNHWC*
strides
*
T0
?
)gradients_4/disc/dense/Relu_grad/ReluGradReluGrad=gradients_4/disc/dense_1/MatMul_grad/tuple/control_dependencydisc/dense/Relu*
T0*(
_output_shapes
:??????????
?
gradients_4/AddN_4AddNJgradients_4/disc/dense_1/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulAgradients_4/disc_1/dense_1/MatMul_grad/tuple/control_dependency_1?gradients_4/disc/dense_1/MatMul_grad/tuple/control_dependency_1*]
_classS
QOloc:@gradients_4/disc/dense_1/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*
T0*
N*
_output_shapes
:	?
?
.gradients_4/disc_1/conv2d_1/Relu_grad/ReluGradReluGrad-gradients_4/disc_1/MaxPool_1_grad/MaxPoolGraddisc_1/conv2d_1/Relu*/
_output_shapes
:?????????2*
T0
?
/gradients_4/disc/dense/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_4/disc/dense/Relu_grad/ReluGrad*
_output_shapes	
:?*
T0*
data_formatNHWC
?
4gradients_4/disc/dense/BiasAdd_grad/tuple/group_depsNoOp0^gradients_4/disc/dense/BiasAdd_grad/BiasAddGrad*^gradients_4/disc/dense/Relu_grad/ReluGrad
?
<gradients_4/disc/dense/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_4/disc/dense/Relu_grad/ReluGrad5^gradients_4/disc/dense/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_4/disc/dense/Relu_grad/ReluGrad*(
_output_shapes
:??????????
?
>gradients_4/disc/dense/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_4/disc/dense/BiasAdd_grad/BiasAddGrad5^gradients_4/disc/dense/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@gradients_4/disc/dense/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:?
?
4gradients_4/disc_1/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients_4/disc_1/conv2d_1/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:2
?
9gradients_4/disc_1/conv2d_1/BiasAdd_grad/tuple/group_depsNoOp5^gradients_4/disc_1/conv2d_1/BiasAdd_grad/BiasAddGrad/^gradients_4/disc_1/conv2d_1/Relu_grad/ReluGrad
?
Agradients_4/disc_1/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients_4/disc_1/conv2d_1/Relu_grad/ReluGrad:^gradients_4/disc_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_4/disc_1/conv2d_1/Relu_grad/ReluGrad*/
_output_shapes
:?????????2
?
Cgradients_4/disc_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_4/disc_1/conv2d_1/BiasAdd_grad/BiasAddGrad:^gradients_4/disc_1/conv2d_1/BiasAdd_grad/tuple/group_deps*G
_class=
;9loc:@gradients_4/disc_1/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:2
?
)gradients_4/disc/dense/MatMul_grad/MatMulMatMul<gradients_4/disc/dense/BiasAdd_grad/tuple/control_dependencydisc/dense/kernel/read*
transpose_b(*(
_output_shapes
:??????????*
T0*
transpose_a( 
?
+gradients_4/disc/dense/MatMul_grad/MatMul_1MatMuldisc/flatten/Reshape<gradients_4/disc/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
??
?
3gradients_4/disc/dense/MatMul_grad/tuple/group_depsNoOp*^gradients_4/disc/dense/MatMul_grad/MatMul,^gradients_4/disc/dense/MatMul_grad/MatMul_1
?
;gradients_4/disc/dense/MatMul_grad/tuple/control_dependencyIdentity)gradients_4/disc/dense/MatMul_grad/MatMul4^gradients_4/disc/dense/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:??????????*<
_class2
0.loc:@gradients_4/disc/dense/MatMul_grad/MatMul
?
=gradients_4/disc/dense/MatMul_grad/tuple/control_dependency_1Identity+gradients_4/disc/dense/MatMul_grad/MatMul_14^gradients_4/disc/dense/MatMul_grad/tuple/group_deps*>
_class4
20loc:@gradients_4/disc/dense/MatMul_grad/MatMul_1* 
_output_shapes
:
??*
T0
?
gradients_4/AddN_5AddN@gradients_4/disc_1/dense/BiasAdd_grad/tuple/control_dependency_1>gradients_4/disc/dense/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:?*
T0*D
_class:
86loc:@gradients_4/disc_1/dense/BiasAdd_grad/BiasAddGrad*
N
?
.gradients_4/disc_1/conv2d_1/Conv2D_grad/ShapeNShapeNdisc_1/MaxPooldisc/conv2d_1/kernel/read* 
_output_shapes
::*
T0*
out_type0*
N
?
;gradients_4/disc_1/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput.gradients_4/disc_1/conv2d_1/Conv2D_grad/ShapeNdisc/conv2d_1/kernel/readAgradients_4/disc_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
use_cudnn_on_gpu(*
	dilations
*
paddingVALID*
explicit_paddings
 *
T0*
strides
*/
_output_shapes
:?????????
?
<gradients_4/disc_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdisc_1/MaxPool0gradients_4/disc_1/conv2d_1/Conv2D_grad/ShapeN:1Agradients_4/disc_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
explicit_paddings
 *
	dilations
*&
_output_shapes
:2*
T0*
data_formatNHWC
?
8gradients_4/disc_1/conv2d_1/Conv2D_grad/tuple/group_depsNoOp=^gradients_4/disc_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilter<^gradients_4/disc_1/conv2d_1/Conv2D_grad/Conv2DBackpropInput
?
@gradients_4/disc_1/conv2d_1/Conv2D_grad/tuple/control_dependencyIdentity;gradients_4/disc_1/conv2d_1/Conv2D_grad/Conv2DBackpropInput9^gradients_4/disc_1/conv2d_1/Conv2D_grad/tuple/group_deps*N
_classD
B@loc:@gradients_4/disc_1/conv2d_1/Conv2D_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:?????????
?
Bgradients_4/disc_1/conv2d_1/Conv2D_grad/tuple/control_dependency_1Identity<gradients_4/disc_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilter9^gradients_4/disc_1/conv2d_1/Conv2D_grad/tuple/group_deps*
T0*&
_output_shapes
:2*O
_classE
CAloc:@gradients_4/disc_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilter
y
+gradients_4/disc/flatten/Reshape_grad/ShapeShapedisc/MaxPool_1*
_output_shapes
:*
out_type0*
T0
?
-gradients_4/disc/flatten/Reshape_grad/ReshapeReshape;gradients_4/disc/dense/MatMul_grad/tuple/control_dependency+gradients_4/disc/flatten/Reshape_grad/Shape*/
_output_shapes
:?????????2*
T0*
Tshape0
?
gradients_4/AddN_6AddNHgradients_4/disc/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul?gradients_4/disc_1/dense/MatMul_grad/tuple/control_dependency_1=gradients_4/disc/dense/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
??*[
_classQ
OMloc:@gradients_4/disc/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*
T0*
N
?
+gradients_4/disc_1/MaxPool_grad/MaxPoolGradMaxPoolGraddisc_1/conv2d/Reludisc_1/MaxPool@gradients_4/disc_1/conv2d_1/Conv2D_grad/tuple/control_dependency*
T0*/
_output_shapes
:?????????*
strides
*
data_formatNHWC*
ksize
*
paddingSAME
?
+gradients_4/disc/MaxPool_1_grad/MaxPoolGradMaxPoolGraddisc/conv2d_1/Reludisc/MaxPool_1-gradients_4/disc/flatten/Reshape_grad/Reshape*
ksize
*
data_formatNHWC*
strides
*
paddingSAME*
T0*/
_output_shapes
:?????????2
?
,gradients_4/disc_1/conv2d/Relu_grad/ReluGradReluGrad+gradients_4/disc_1/MaxPool_grad/MaxPoolGraddisc_1/conv2d/Relu*/
_output_shapes
:?????????*
T0
?
,gradients_4/disc/conv2d_1/Relu_grad/ReluGradReluGrad+gradients_4/disc/MaxPool_1_grad/MaxPoolGraddisc/conv2d_1/Relu*
T0*/
_output_shapes
:?????????2
?
2gradients_4/disc_1/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_4/disc_1/conv2d/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
?
7gradients_4/disc_1/conv2d/BiasAdd_grad/tuple/group_depsNoOp3^gradients_4/disc_1/conv2d/BiasAdd_grad/BiasAddGrad-^gradients_4/disc_1/conv2d/Relu_grad/ReluGrad
?
?gradients_4/disc_1/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity,gradients_4/disc_1/conv2d/Relu_grad/ReluGrad8^gradients_4/disc_1/conv2d/BiasAdd_grad/tuple/group_deps*
T0*/
_output_shapes
:?????????*?
_class5
31loc:@gradients_4/disc_1/conv2d/Relu_grad/ReluGrad
?
Agradients_4/disc_1/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_4/disc_1/conv2d/BiasAdd_grad/BiasAddGrad8^gradients_4/disc_1/conv2d/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*E
_class;
97loc:@gradients_4/disc_1/conv2d/BiasAdd_grad/BiasAddGrad
?
2gradients_4/disc/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_4/disc/conv2d_1/Relu_grad/ReluGrad*
_output_shapes
:2*
data_formatNHWC*
T0
?
7gradients_4/disc/conv2d_1/BiasAdd_grad/tuple/group_depsNoOp3^gradients_4/disc/conv2d_1/BiasAdd_grad/BiasAddGrad-^gradients_4/disc/conv2d_1/Relu_grad/ReluGrad
?
?gradients_4/disc/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity,gradients_4/disc/conv2d_1/Relu_grad/ReluGrad8^gradients_4/disc/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*/
_output_shapes
:?????????2*?
_class5
31loc:@gradients_4/disc/conv2d_1/Relu_grad/ReluGrad
?
Agradients_4/disc/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_4/disc/conv2d_1/BiasAdd_grad/BiasAddGrad8^gradients_4/disc/conv2d_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:2*E
_class;
97loc:@gradients_4/disc/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0
?
,gradients_4/disc_1/conv2d/Conv2D_grad/ShapeNShapeNdisc_1/Reshapedisc/conv2d/kernel/read*
T0* 
_output_shapes
::*
N*
out_type0
?
9gradients_4/disc_1/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput,gradients_4/disc_1/conv2d/Conv2D_grad/ShapeNdisc/conv2d/kernel/read?gradients_4/disc_1/conv2d/BiasAdd_grad/tuple/control_dependency*
	dilations
*
data_formatNHWC*
strides
*
T0*
explicit_paddings
 */
_output_shapes
:?????????*
paddingVALID*
use_cudnn_on_gpu(
?
:gradients_4/disc_1/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdisc_1/Reshape.gradients_4/disc_1/conv2d/Conv2D_grad/ShapeN:1?gradients_4/disc_1/conv2d/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*&
_output_shapes
:*
	dilations
*
T0*
paddingVALID*
data_formatNHWC*
explicit_paddings
 *
strides

?
6gradients_4/disc_1/conv2d/Conv2D_grad/tuple/group_depsNoOp;^gradients_4/disc_1/conv2d/Conv2D_grad/Conv2DBackpropFilter:^gradients_4/disc_1/conv2d/Conv2D_grad/Conv2DBackpropInput
?
>gradients_4/disc_1/conv2d/Conv2D_grad/tuple/control_dependencyIdentity9gradients_4/disc_1/conv2d/Conv2D_grad/Conv2DBackpropInput7^gradients_4/disc_1/conv2d/Conv2D_grad/tuple/group_deps*/
_output_shapes
:?????????*L
_classB
@>loc:@gradients_4/disc_1/conv2d/Conv2D_grad/Conv2DBackpropInput*
T0
?
@gradients_4/disc_1/conv2d/Conv2D_grad/tuple/control_dependency_1Identity:gradients_4/disc_1/conv2d/Conv2D_grad/Conv2DBackpropFilter7^gradients_4/disc_1/conv2d/Conv2D_grad/tuple/group_deps*M
_classC
A?loc:@gradients_4/disc_1/conv2d/Conv2D_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
?
,gradients_4/disc/conv2d_1/Conv2D_grad/ShapeNShapeNdisc/MaxPooldisc/conv2d_1/kernel/read* 
_output_shapes
::*
N*
out_type0*
T0
?
9gradients_4/disc/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput,gradients_4/disc/conv2d_1/Conv2D_grad/ShapeNdisc/conv2d_1/kernel/read?gradients_4/disc/conv2d_1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
strides
*
data_formatNHWC*
T0*
paddingVALID*/
_output_shapes
:?????????*
explicit_paddings
 *
use_cudnn_on_gpu(
?
:gradients_4/disc/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdisc/MaxPool.gradients_4/disc/conv2d_1/Conv2D_grad/ShapeN:1?gradients_4/disc/conv2d_1/BiasAdd_grad/tuple/control_dependency*
T0*
	dilations
*&
_output_shapes
:2*
paddingVALID*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
data_formatNHWC
?
6gradients_4/disc/conv2d_1/Conv2D_grad/tuple/group_depsNoOp;^gradients_4/disc/conv2d_1/Conv2D_grad/Conv2DBackpropFilter:^gradients_4/disc/conv2d_1/Conv2D_grad/Conv2DBackpropInput
?
>gradients_4/disc/conv2d_1/Conv2D_grad/tuple/control_dependencyIdentity9gradients_4/disc/conv2d_1/Conv2D_grad/Conv2DBackpropInput7^gradients_4/disc/conv2d_1/Conv2D_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_4/disc/conv2d_1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????
?
@gradients_4/disc/conv2d_1/Conv2D_grad/tuple/control_dependency_1Identity:gradients_4/disc/conv2d_1/Conv2D_grad/Conv2DBackpropFilter7^gradients_4/disc/conv2d_1/Conv2D_grad/tuple/group_deps*M
_classC
A?loc:@gradients_4/disc/conv2d_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:2*
T0
?
gradients_4/AddN_7AddNCgradients_4/disc_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Agradients_4/disc/conv2d_1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:2*G
_class=
;9loc:@gradients_4/disc_1/conv2d_1/BiasAdd_grad/BiasAddGrad*
N
?
)gradients_4/disc/MaxPool_grad/MaxPoolGradMaxPoolGraddisc/conv2d/Reludisc/MaxPool>gradients_4/disc/conv2d_1/Conv2D_grad/tuple/control_dependency*
ksize
*/
_output_shapes
:?????????*
paddingSAME*
strides
*
T0*
data_formatNHWC
?
gradients_4/AddN_8AddNKgradients_4/disc/conv2d_1/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulBgradients_4/disc_1/conv2d_1/Conv2D_grad/tuple/control_dependency_1@gradients_4/disc/conv2d_1/Conv2D_grad/tuple/control_dependency_1*^
_classT
RPloc:@gradients_4/disc/conv2d_1/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*
T0*
N*&
_output_shapes
:2
?
*gradients_4/disc/conv2d/Relu_grad/ReluGradReluGrad)gradients_4/disc/MaxPool_grad/MaxPoolGraddisc/conv2d/Relu*
T0*/
_output_shapes
:?????????
?
0gradients_4/disc/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients_4/disc/conv2d/Relu_grad/ReluGrad*
_output_shapes
:*
data_formatNHWC*
T0
?
5gradients_4/disc/conv2d/BiasAdd_grad/tuple/group_depsNoOp1^gradients_4/disc/conv2d/BiasAdd_grad/BiasAddGrad+^gradients_4/disc/conv2d/Relu_grad/ReluGrad
?
=gradients_4/disc/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity*gradients_4/disc/conv2d/Relu_grad/ReluGrad6^gradients_4/disc/conv2d/BiasAdd_grad/tuple/group_deps*
T0*/
_output_shapes
:?????????*=
_class3
1/loc:@gradients_4/disc/conv2d/Relu_grad/ReluGrad
?
?gradients_4/disc/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity0gradients_4/disc/conv2d/BiasAdd_grad/BiasAddGrad6^gradients_4/disc/conv2d/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@gradients_4/disc/conv2d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
?
*gradients_4/disc/conv2d/Conv2D_grad/ShapeNShapeNdisc/Reshapedisc/conv2d/kernel/read*
out_type0*
T0*
N* 
_output_shapes
::
?
7gradients_4/disc/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients_4/disc/conv2d/Conv2D_grad/ShapeNdisc/conv2d/kernel/read=gradients_4/disc/conv2d/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
	dilations
*
data_formatNHWC*
explicit_paddings
 */
_output_shapes
:?????????*
T0
?
8gradients_4/disc/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdisc/Reshape,gradients_4/disc/conv2d/Conv2D_grad/ShapeN:1=gradients_4/disc/conv2d/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
data_formatNHWC*
	dilations
*&
_output_shapes
:*
T0*
use_cudnn_on_gpu(*
strides
*
explicit_paddings
 
?
4gradients_4/disc/conv2d/Conv2D_grad/tuple/group_depsNoOp9^gradients_4/disc/conv2d/Conv2D_grad/Conv2DBackpropFilter8^gradients_4/disc/conv2d/Conv2D_grad/Conv2DBackpropInput
?
<gradients_4/disc/conv2d/Conv2D_grad/tuple/control_dependencyIdentity7gradients_4/disc/conv2d/Conv2D_grad/Conv2DBackpropInput5^gradients_4/disc/conv2d/Conv2D_grad/tuple/group_deps*J
_class@
><loc:@gradients_4/disc/conv2d/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????*
T0
?
>gradients_4/disc/conv2d/Conv2D_grad/tuple/control_dependency_1Identity8gradients_4/disc/conv2d/Conv2D_grad/Conv2DBackpropFilter5^gradients_4/disc/conv2d/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_4/disc/conv2d/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
?
gradients_4/AddN_9AddNAgradients_4/disc_1/conv2d/BiasAdd_grad/tuple/control_dependency_1?gradients_4/disc/conv2d/BiasAdd_grad/tuple/control_dependency_1*
T0*
N*
_output_shapes
:*E
_class;
97loc:@gradients_4/disc_1/conv2d/BiasAdd_grad/BiasAddGrad
?
gradients_4/AddN_10AddNIgradients_4/disc/conv2d/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul@gradients_4/disc_1/conv2d/Conv2D_grad/tuple/control_dependency_1>gradients_4/disc/conv2d/Conv2D_grad/tuple/control_dependency_1*
N*\
_classR
PNloc:@gradients_4/disc/conv2d/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*&
_output_shapes
:*
T0
?
beta1_power_1/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *fff?*#
_class
loc:@disc/conv2d/bias
?
beta1_power_1
VariableV2*
shape: *
_output_shapes
: *#
_class
loc:@disc/conv2d/bias*
dtype0*
	container *
shared_name 
?
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
validate_shape(*
use_locking(*
_output_shapes
: *#
_class
loc:@disc/conv2d/bias*
T0
s
beta1_power_1/readIdentitybeta1_power_1*
_output_shapes
: *#
_class
loc:@disc/conv2d/bias*
T0
?
beta2_power_1/initial_valueConst*#
_class
loc:@disc/conv2d/bias*
_output_shapes
: *
valueB
 *w??*
dtype0
?
beta2_power_1
VariableV2*
_output_shapes
: *#
_class
loc:@disc/conv2d/bias*
shape: *
dtype0*
shared_name *
	container 
?
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
_output_shapes
: *
T0*
validate_shape(*#
_class
loc:@disc/conv2d/bias*
use_locking(
s
beta2_power_1/readIdentitybeta2_power_1*
T0*
_output_shapes
: *#
_class
loc:@disc/conv2d/bias
?
)disc/conv2d/kernel/Adam/Initializer/zerosConst*
dtype0*%
valueB*    *%
_class
loc:@disc/conv2d/kernel*&
_output_shapes
:
?
disc/conv2d/kernel/Adam
VariableV2*
dtype0*
	container *%
_class
loc:@disc/conv2d/kernel*
shape:*
shared_name *&
_output_shapes
:
?
disc/conv2d/kernel/Adam/AssignAssigndisc/conv2d/kernel/Adam)disc/conv2d/kernel/Adam/Initializer/zeros*
validate_shape(*%
_class
loc:@disc/conv2d/kernel*
T0*&
_output_shapes
:*
use_locking(
?
disc/conv2d/kernel/Adam/readIdentitydisc/conv2d/kernel/Adam*&
_output_shapes
:*
T0*%
_class
loc:@disc/conv2d/kernel
?
+disc/conv2d/kernel/Adam_1/Initializer/zerosConst*%
_class
loc:@disc/conv2d/kernel*%
valueB*    *
dtype0*&
_output_shapes
:
?
disc/conv2d/kernel/Adam_1
VariableV2*
dtype0*
shape:*%
_class
loc:@disc/conv2d/kernel*
shared_name *
	container *&
_output_shapes
:
?
 disc/conv2d/kernel/Adam_1/AssignAssigndisc/conv2d/kernel/Adam_1+disc/conv2d/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@disc/conv2d/kernel*
validate_shape(*&
_output_shapes
:
?
disc/conv2d/kernel/Adam_1/readIdentitydisc/conv2d/kernel/Adam_1*&
_output_shapes
:*
T0*%
_class
loc:@disc/conv2d/kernel
?
'disc/conv2d/bias/Adam/Initializer/zerosConst*#
_class
loc:@disc/conv2d/bias*
dtype0*
valueB*    *
_output_shapes
:
?
disc/conv2d/bias/Adam
VariableV2*
dtype0*
shape:*
shared_name *
_output_shapes
:*
	container *#
_class
loc:@disc/conv2d/bias
?
disc/conv2d/bias/Adam/AssignAssigndisc/conv2d/bias/Adam'disc/conv2d/bias/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_output_shapes
:*#
_class
loc:@disc/conv2d/bias
?
disc/conv2d/bias/Adam/readIdentitydisc/conv2d/bias/Adam*
_output_shapes
:*#
_class
loc:@disc/conv2d/bias*
T0
?
)disc/conv2d/bias/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*#
_class
loc:@disc/conv2d/bias
?
disc/conv2d/bias/Adam_1
VariableV2*
_output_shapes
:*
dtype0*
shape:*
shared_name *#
_class
loc:@disc/conv2d/bias*
	container 
?
disc/conv2d/bias/Adam_1/AssignAssigndisc/conv2d/bias/Adam_1)disc/conv2d/bias/Adam_1/Initializer/zeros*
T0*#
_class
loc:@disc/conv2d/bias*
_output_shapes
:*
validate_shape(*
use_locking(
?
disc/conv2d/bias/Adam_1/readIdentitydisc/conv2d/bias/Adam_1*
_output_shapes
:*
T0*#
_class
loc:@disc/conv2d/bias
?
;disc/conv2d_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@disc/conv2d_1/kernel*%
valueB"         2   *
_output_shapes
:*
dtype0
?
1disc/conv2d_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *'
_class
loc:@disc/conv2d_1/kernel*
_output_shapes
: *
dtype0
?
+disc/conv2d_1/kernel/Adam/Initializer/zerosFill;disc/conv2d_1/kernel/Adam/Initializer/zeros/shape_as_tensor1disc/conv2d_1/kernel/Adam/Initializer/zeros/Const*

index_type0*&
_output_shapes
:2*
T0*'
_class
loc:@disc/conv2d_1/kernel
?
disc/conv2d_1/kernel/Adam
VariableV2*
shape:2*
	container *&
_output_shapes
:2*
dtype0*'
_class
loc:@disc/conv2d_1/kernel*
shared_name 
?
 disc/conv2d_1/kernel/Adam/AssignAssigndisc/conv2d_1/kernel/Adam+disc/conv2d_1/kernel/Adam/Initializer/zeros*
use_locking(*&
_output_shapes
:2*'
_class
loc:@disc/conv2d_1/kernel*
T0*
validate_shape(
?
disc/conv2d_1/kernel/Adam/readIdentitydisc/conv2d_1/kernel/Adam*
T0*'
_class
loc:@disc/conv2d_1/kernel*&
_output_shapes
:2
?
=disc/conv2d_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@disc/conv2d_1/kernel*
dtype0*
_output_shapes
:*%
valueB"         2   
?
3disc/conv2d_1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *'
_class
loc:@disc/conv2d_1/kernel*
dtype0
?
-disc/conv2d_1/kernel/Adam_1/Initializer/zerosFill=disc/conv2d_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor3disc/conv2d_1/kernel/Adam_1/Initializer/zeros/Const*

index_type0*
T0*'
_class
loc:@disc/conv2d_1/kernel*&
_output_shapes
:2
?
disc/conv2d_1/kernel/Adam_1
VariableV2*&
_output_shapes
:2*
shape:2*
shared_name *
	container *
dtype0*'
_class
loc:@disc/conv2d_1/kernel
?
"disc/conv2d_1/kernel/Adam_1/AssignAssigndisc/conv2d_1/kernel/Adam_1-disc/conv2d_1/kernel/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
:2*
use_locking(*
T0*'
_class
loc:@disc/conv2d_1/kernel
?
 disc/conv2d_1/kernel/Adam_1/readIdentitydisc/conv2d_1/kernel/Adam_1*
T0*&
_output_shapes
:2*'
_class
loc:@disc/conv2d_1/kernel
?
)disc/conv2d_1/bias/Adam/Initializer/zerosConst*
dtype0*
valueB2*    *%
_class
loc:@disc/conv2d_1/bias*
_output_shapes
:2
?
disc/conv2d_1/bias/Adam
VariableV2*
_output_shapes
:2*
shared_name *
shape:2*
dtype0*
	container *%
_class
loc:@disc/conv2d_1/bias
?
disc/conv2d_1/bias/Adam/AssignAssigndisc/conv2d_1/bias/Adam)disc/conv2d_1/bias/Adam/Initializer/zeros*
T0*
use_locking(*%
_class
loc:@disc/conv2d_1/bias*
_output_shapes
:2*
validate_shape(
?
disc/conv2d_1/bias/Adam/readIdentitydisc/conv2d_1/bias/Adam*
_output_shapes
:2*%
_class
loc:@disc/conv2d_1/bias*
T0
?
+disc/conv2d_1/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:2*%
_class
loc:@disc/conv2d_1/bias*
dtype0*
valueB2*    
?
disc/conv2d_1/bias/Adam_1
VariableV2*
shape:2*
shared_name *
dtype0*
_output_shapes
:2*%
_class
loc:@disc/conv2d_1/bias*
	container 
?
 disc/conv2d_1/bias/Adam_1/AssignAssigndisc/conv2d_1/bias/Adam_1+disc/conv2d_1/bias/Adam_1/Initializer/zeros*
T0*%
_class
loc:@disc/conv2d_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:2
?
disc/conv2d_1/bias/Adam_1/readIdentitydisc/conv2d_1/bias/Adam_1*
_output_shapes
:2*%
_class
loc:@disc/conv2d_1/bias*
T0
?
8disc/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*$
_class
loc:@disc/dense/kernel*
valueB"   ?  
?
.disc/dense/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0*$
_class
loc:@disc/dense/kernel
?
(disc/dense/kernel/Adam/Initializer/zerosFill8disc/dense/kernel/Adam/Initializer/zeros/shape_as_tensor.disc/dense/kernel/Adam/Initializer/zeros/Const*

index_type0*$
_class
loc:@disc/dense/kernel* 
_output_shapes
:
??*
T0
?
disc/dense/kernel/Adam
VariableV2*
dtype0*
	container *
shared_name *$
_class
loc:@disc/dense/kernel* 
_output_shapes
:
??*
shape:
??
?
disc/dense/kernel/Adam/AssignAssigndisc/dense/kernel/Adam(disc/dense/kernel/Adam/Initializer/zeros* 
_output_shapes
:
??*$
_class
loc:@disc/dense/kernel*
use_locking(*
T0*
validate_shape(
?
disc/dense/kernel/Adam/readIdentitydisc/dense/kernel/Adam*
T0* 
_output_shapes
:
??*$
_class
loc:@disc/dense/kernel
?
:disc/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@disc/dense/kernel*
_output_shapes
:*
valueB"   ?  *
dtype0
?
0disc/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *$
_class
loc:@disc/dense/kernel
?
*disc/dense/kernel/Adam_1/Initializer/zerosFill:disc/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor0disc/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@disc/dense/kernel* 
_output_shapes
:
??
?
disc/dense/kernel/Adam_1
VariableV2*
	container *
shape:
??*
shared_name *$
_class
loc:@disc/dense/kernel* 
_output_shapes
:
??*
dtype0
?
disc/dense/kernel/Adam_1/AssignAssigndisc/dense/kernel/Adam_1*disc/dense/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
??*$
_class
loc:@disc/dense/kernel*
use_locking(*
validate_shape(*
T0
?
disc/dense/kernel/Adam_1/readIdentitydisc/dense/kernel/Adam_1* 
_output_shapes
:
??*$
_class
loc:@disc/dense/kernel*
T0
?
&disc/dense/bias/Adam/Initializer/zerosConst*
valueB?*    *"
_class
loc:@disc/dense/bias*
dtype0*
_output_shapes	
:?
?
disc/dense/bias/Adam
VariableV2*"
_class
loc:@disc/dense/bias*
shared_name *
shape:?*
dtype0*
	container *
_output_shapes	
:?
?
disc/dense/bias/Adam/AssignAssigndisc/dense/bias/Adam&disc/dense/bias/Adam/Initializer/zeros*"
_class
loc:@disc/dense/bias*
T0*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
disc/dense/bias/Adam/readIdentitydisc/dense/bias/Adam*
T0*"
_class
loc:@disc/dense/bias*
_output_shapes	
:?
?
(disc/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *"
_class
loc:@disc/dense/bias*
dtype0
?
disc/dense/bias/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes	
:?*
shape:?*"
_class
loc:@disc/dense/bias
?
disc/dense/bias/Adam_1/AssignAssigndisc/dense/bias/Adam_1(disc/dense/bias/Adam_1/Initializer/zeros*"
_class
loc:@disc/dense/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(*
T0
?
disc/dense/bias/Adam_1/readIdentitydisc/dense/bias/Adam_1*
T0*"
_class
loc:@disc/dense/bias*
_output_shapes	
:?
?
:disc/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"?     *&
_class
loc:@disc/dense_1/kernel*
_output_shapes
:*
dtype0
?
0disc/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *&
_class
loc:@disc/dense_1/kernel*
dtype0*
_output_shapes
: 
?
*disc/dense_1/kernel/Adam/Initializer/zerosFill:disc/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor0disc/dense_1/kernel/Adam/Initializer/zeros/Const*&
_class
loc:@disc/dense_1/kernel*

index_type0*
T0*
_output_shapes
:	?
?
disc/dense_1/kernel/Adam
VariableV2*
shared_name *&
_class
loc:@disc/dense_1/kernel*
	container *
dtype0*
shape:	?*
_output_shapes
:	?
?
disc/dense_1/kernel/Adam/AssignAssigndisc/dense_1/kernel/Adam*disc/dense_1/kernel/Adam/Initializer/zeros*&
_class
loc:@disc/dense_1/kernel*
validate_shape(*
_output_shapes
:	?*
T0*
use_locking(
?
disc/dense_1/kernel/Adam/readIdentitydisc/dense_1/kernel/Adam*
T0*
_output_shapes
:	?*&
_class
loc:@disc/dense_1/kernel
?
<disc/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"?     *
dtype0*&
_class
loc:@disc/dense_1/kernel
?
2disc/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*&
_class
loc:@disc/dense_1/kernel*
_output_shapes
: *
valueB
 *    *
dtype0
?
,disc/dense_1/kernel/Adam_1/Initializer/zerosFill<disc/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor2disc/dense_1/kernel/Adam_1/Initializer/zeros/Const*&
_class
loc:@disc/dense_1/kernel*

index_type0*
T0*
_output_shapes
:	?
?
disc/dense_1/kernel/Adam_1
VariableV2*
	container *
shared_name *
_output_shapes
:	?*
shape:	?*&
_class
loc:@disc/dense_1/kernel*
dtype0
?
!disc/dense_1/kernel/Adam_1/AssignAssigndisc/dense_1/kernel/Adam_1,disc/dense_1/kernel/Adam_1/Initializer/zeros*
validate_shape(*&
_class
loc:@disc/dense_1/kernel*
_output_shapes
:	?*
use_locking(*
T0
?
disc/dense_1/kernel/Adam_1/readIdentitydisc/dense_1/kernel/Adam_1*&
_class
loc:@disc/dense_1/kernel*
_output_shapes
:	?*
T0
?
(disc/dense_1/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*$
_class
loc:@disc/dense_1/bias
?
disc/dense_1/bias/Adam
VariableV2*
shape:*
	container *
_output_shapes
:*
dtype0*
shared_name *$
_class
loc:@disc/dense_1/bias
?
disc/dense_1/bias/Adam/AssignAssigndisc/dense_1/bias/Adam(disc/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*$
_class
loc:@disc/dense_1/bias
?
disc/dense_1/bias/Adam/readIdentitydisc/dense_1/bias/Adam*$
_class
loc:@disc/dense_1/bias*
T0*
_output_shapes
:
?
*disc/dense_1/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*
valueB*    *$
_class
loc:@disc/dense_1/bias*
dtype0
?
disc/dense_1/bias/Adam_1
VariableV2*
_output_shapes
:*
shape:*
	container *
dtype0*$
_class
loc:@disc/dense_1/bias*
shared_name 
?
disc/dense_1/bias/Adam_1/AssignAssigndisc/dense_1/bias/Adam_1*disc/dense_1/bias/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*$
_class
loc:@disc/dense_1/bias
?
disc/dense_1/bias/Adam_1/readIdentitydisc/dense_1/bias/Adam_1*$
_class
loc:@disc/dense_1/bias*
_output_shapes
:*
T0
Y
Adam_1/learning_rateConst*
dtype0*
valueB
 *o?:*
_output_shapes
: 
Q
Adam_1/beta1Const*
valueB
 *fff?*
_output_shapes
: *
dtype0
Q
Adam_1/beta2Const*
valueB
 *w??*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *w?+2*
_output_shapes
: *
dtype0
?
*Adam_1/update_disc/conv2d/kernel/ApplyAdam	ApplyAdamdisc/conv2d/kerneldisc/conv2d/kernel/Adamdisc/conv2d/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_4/AddN_10*
use_locking( *
use_nesterov( *
T0*&
_output_shapes
:*%
_class
loc:@disc/conv2d/kernel
?
(Adam_1/update_disc/conv2d/bias/ApplyAdam	ApplyAdamdisc/conv2d/biasdisc/conv2d/bias/Adamdisc/conv2d/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_4/AddN_9*
use_locking( *
T0*
use_nesterov( *
_output_shapes
:*#
_class
loc:@disc/conv2d/bias
?
,Adam_1/update_disc/conv2d_1/kernel/ApplyAdam	ApplyAdamdisc/conv2d_1/kerneldisc/conv2d_1/kernel/Adamdisc/conv2d_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_4/AddN_8*
use_nesterov( *
use_locking( *'
_class
loc:@disc/conv2d_1/kernel*&
_output_shapes
:2*
T0
?
*Adam_1/update_disc/conv2d_1/bias/ApplyAdam	ApplyAdamdisc/conv2d_1/biasdisc/conv2d_1/bias/Adamdisc/conv2d_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_4/AddN_7*%
_class
loc:@disc/conv2d_1/bias*
_output_shapes
:2*
use_nesterov( *
T0*
use_locking( 
?
)Adam_1/update_disc/dense/kernel/ApplyAdam	ApplyAdamdisc/dense/kerneldisc/dense/kernel/Adamdisc/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_4/AddN_6*$
_class
loc:@disc/dense/kernel*
use_locking( *
use_nesterov( *
T0* 
_output_shapes
:
??
?
'Adam_1/update_disc/dense/bias/ApplyAdam	ApplyAdamdisc/dense/biasdisc/dense/bias/Adamdisc/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_4/AddN_5*
use_locking( *
T0*
use_nesterov( *
_output_shapes	
:?*"
_class
loc:@disc/dense/bias
?
+Adam_1/update_disc/dense_1/kernel/ApplyAdam	ApplyAdamdisc/dense_1/kerneldisc/dense_1/kernel/Adamdisc/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_4/AddN_4*
use_locking( *
_output_shapes
:	?*&
_class
loc:@disc/dense_1/kernel*
T0*
use_nesterov( 
?
)Adam_1/update_disc/dense_1/bias/ApplyAdam	ApplyAdamdisc/dense_1/biasdisc/dense_1/bias/Adamdisc/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_4/AddN_3*
_output_shapes
:*
use_nesterov( *$
_class
loc:@disc/dense_1/bias*
use_locking( *
T0
?

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1)^Adam_1/update_disc/conv2d/bias/ApplyAdam+^Adam_1/update_disc/conv2d/kernel/ApplyAdam+^Adam_1/update_disc/conv2d_1/bias/ApplyAdam-^Adam_1/update_disc/conv2d_1/kernel/ApplyAdam(^Adam_1/update_disc/dense/bias/ApplyAdam*^Adam_1/update_disc/dense/kernel/ApplyAdam*^Adam_1/update_disc/dense_1/bias/ApplyAdam,^Adam_1/update_disc/dense_1/kernel/ApplyAdam*
_output_shapes
: *#
_class
loc:@disc/conv2d/bias*
T0
?
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
T0*
use_locking( *
validate_shape(*#
_class
loc:@disc/conv2d/bias*
_output_shapes
: 
?
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2)^Adam_1/update_disc/conv2d/bias/ApplyAdam+^Adam_1/update_disc/conv2d/kernel/ApplyAdam+^Adam_1/update_disc/conv2d_1/bias/ApplyAdam-^Adam_1/update_disc/conv2d_1/kernel/ApplyAdam(^Adam_1/update_disc/dense/bias/ApplyAdam*^Adam_1/update_disc/dense/kernel/ApplyAdam*^Adam_1/update_disc/dense_1/bias/ApplyAdam,^Adam_1/update_disc/dense_1/kernel/ApplyAdam*#
_class
loc:@disc/conv2d/bias*
_output_shapes
: *
T0
?
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
validate_shape(*
T0*
use_locking( *#
_class
loc:@disc/conv2d/bias*
_output_shapes
: 
?
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1)^Adam_1/update_disc/conv2d/bias/ApplyAdam+^Adam_1/update_disc/conv2d/kernel/ApplyAdam+^Adam_1/update_disc/conv2d_1/bias/ApplyAdam-^Adam_1/update_disc/conv2d_1/kernel/ApplyAdam(^Adam_1/update_disc/dense/bias/ApplyAdam*^Adam_1/update_disc/dense/kernel/ApplyAdam*^Adam_1/update_disc/dense_1/bias/ApplyAdam,^Adam_1/update_disc/dense_1/kernel/ApplyAdam
T
gradients_5/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
Z
gradients_5/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ??
u
gradients_5/FillFillgradients_5/Shapegradients_5/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
p
&gradients_5/Mean_11_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
?
 gradients_5/Mean_11_grad/ReshapeReshapegradients_5/Fill&gradients_5/Mean_11_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
d
gradients_5/Mean_11_grad/ShapeShapeadd_21*
T0*
out_type0*
_output_shapes
:
?
gradients_5/Mean_11_grad/TileTile gradients_5/Mean_11_grad/Reshapegradients_5/Mean_11_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:?????????
f
 gradients_5/Mean_11_grad/Shape_1Shapeadd_21*
T0*
_output_shapes
:*
out_type0
c
 gradients_5/Mean_11_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
h
gradients_5/Mean_11_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
gradients_5/Mean_11_grad/ProdProd gradients_5/Mean_11_grad/Shape_1gradients_5/Mean_11_grad/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
j
 gradients_5/Mean_11_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
?
gradients_5/Mean_11_grad/Prod_1Prod gradients_5/Mean_11_grad/Shape_2 gradients_5/Mean_11_grad/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
d
"gradients_5/Mean_11_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
?
 gradients_5/Mean_11_grad/MaximumMaximumgradients_5/Mean_11_grad/Prod_1"gradients_5/Mean_11_grad/Maximum/y*
T0*
_output_shapes
: 
?
!gradients_5/Mean_11_grad/floordivFloorDivgradients_5/Mean_11_grad/Prod gradients_5/Mean_11_grad/Maximum*
_output_shapes
: *
T0
?
gradients_5/Mean_11_grad/CastCast!gradients_5/Mean_11_grad/floordiv*

DstT0*
Truncate( *
_output_shapes
: *

SrcT0
?
 gradients_5/Mean_11_grad/truedivRealDivgradients_5/Mean_11_grad/Tilegradients_5/Mean_11_grad/Cast*
T0*#
_output_shapes
:?????????
w
&gradients_5/Mean_12_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
?
 gradients_5/Mean_12_grad/ReshapeReshapegradients_5/Fill&gradients_5/Mean_12_grad/Reshape/shape*
T0*
_output_shapes

:*
Tshape0
d
gradients_5/Mean_12_grad/ShapeShapeadd_24*
T0*
out_type0*
_output_shapes
:
?
gradients_5/Mean_12_grad/TileTile gradients_5/Mean_12_grad/Reshapegradients_5/Mean_12_grad/Shape*
T0*

Tmultiples0*'
_output_shapes
:?????????
f
 gradients_5/Mean_12_grad/Shape_1Shapeadd_24*
_output_shapes
:*
out_type0*
T0
c
 gradients_5/Mean_12_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
h
gradients_5/Mean_12_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
?
gradients_5/Mean_12_grad/ProdProd gradients_5/Mean_12_grad/Shape_1gradients_5/Mean_12_grad/Const*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
j
 gradients_5/Mean_12_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
gradients_5/Mean_12_grad/Prod_1Prod gradients_5/Mean_12_grad/Shape_2 gradients_5/Mean_12_grad/Const_1*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0
d
"gradients_5/Mean_12_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
 gradients_5/Mean_12_grad/MaximumMaximumgradients_5/Mean_12_grad/Prod_1"gradients_5/Mean_12_grad/Maximum/y*
_output_shapes
: *
T0
?
!gradients_5/Mean_12_grad/floordivFloorDivgradients_5/Mean_12_grad/Prod gradients_5/Mean_12_grad/Maximum*
T0*
_output_shapes
: 
?
gradients_5/Mean_12_grad/CastCast!gradients_5/Mean_12_grad/floordiv*
_output_shapes
: *

SrcT0*
Truncate( *

DstT0
?
 gradients_5/Mean_12_grad/truedivRealDivgradients_5/Mean_12_grad/Tilegradients_5/Mean_12_grad/Cast*'
_output_shapes
:?????????*
T0
b
gradients_5/add_21_grad/ShapeShapeSum_6*
T0*
out_type0*
_output_shapes
:
d
gradients_5/add_21_grad/Shape_1ShapeSum_7*
T0*
out_type0*
_output_shapes
:
?
-gradients_5/add_21_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/add_21_grad/Shapegradients_5/add_21_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_5/add_21_grad/SumSum gradients_5/Mean_11_grad/truediv-gradients_5/add_21_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
?
gradients_5/add_21_grad/ReshapeReshapegradients_5/add_21_grad/Sumgradients_5/add_21_grad/Shape*#
_output_shapes
:?????????*
Tshape0*
T0
?
gradients_5/add_21_grad/Sum_1Sum gradients_5/Mean_11_grad/truediv/gradients_5/add_21_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
?
!gradients_5/add_21_grad/Reshape_1Reshapegradients_5/add_21_grad/Sum_1gradients_5/add_21_grad/Shape_1*#
_output_shapes
:?????????*
T0*
Tshape0
c
gradients_5/add_24_grad/ShapeShapeadd_23*
out_type0*
T0*
_output_shapes
:
p
gradients_5/add_24_grad/Shape_1Const*
valueB"      *
_output_shapes
:*
dtype0
?
-gradients_5/add_24_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/add_24_grad/Shapegradients_5/add_24_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_5/add_24_grad/SumSum gradients_5/Mean_12_grad/truediv-gradients_5/add_24_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
?
gradients_5/add_24_grad/ReshapeReshapegradients_5/add_24_grad/Sumgradients_5/add_24_grad/Shape*
Tshape0*'
_output_shapes
:?????????*
T0
?
gradients_5/add_24_grad/Sum_1Sum gradients_5/Mean_12_grad/truediv/gradients_5/add_24_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
?
!gradients_5/add_24_grad/Reshape_1Reshapegradients_5/add_24_grad/Sum_1gradients_5/add_24_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0
a
gradients_5/Sum_6_grad/ShapeShapemul_3*
out_type0*
_output_shapes
:*
T0
?
gradients_5/Sum_6_grad/SizeConst*
_output_shapes
: *
value	B :*
dtype0*/
_class%
#!loc:@gradients_5/Sum_6_grad/Shape
?
gradients_5/Sum_6_grad/addAddSum_6/reduction_indicesgradients_5/Sum_6_grad/Size*
_output_shapes
: */
_class%
#!loc:@gradients_5/Sum_6_grad/Shape*
T0
?
gradients_5/Sum_6_grad/modFloorModgradients_5/Sum_6_grad/addgradients_5/Sum_6_grad/Size*
_output_shapes
: */
_class%
#!loc:@gradients_5/Sum_6_grad/Shape*
T0
?
gradients_5/Sum_6_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB */
_class%
#!loc:@gradients_5/Sum_6_grad/Shape
?
"gradients_5/Sum_6_grad/range/startConst*/
_class%
#!loc:@gradients_5/Sum_6_grad/Shape*
_output_shapes
: *
dtype0*
value	B : 
?
"gradients_5/Sum_6_grad/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: */
_class%
#!loc:@gradients_5/Sum_6_grad/Shape
?
gradients_5/Sum_6_grad/rangeRange"gradients_5/Sum_6_grad/range/startgradients_5/Sum_6_grad/Size"gradients_5/Sum_6_grad/range/delta*
_output_shapes
:*

Tidx0*/
_class%
#!loc:@gradients_5/Sum_6_grad/Shape
?
!gradients_5/Sum_6_grad/Fill/valueConst*/
_class%
#!loc:@gradients_5/Sum_6_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
gradients_5/Sum_6_grad/FillFillgradients_5/Sum_6_grad/Shape_1!gradients_5/Sum_6_grad/Fill/value*/
_class%
#!loc:@gradients_5/Sum_6_grad/Shape*
T0*
_output_shapes
: *

index_type0
?
$gradients_5/Sum_6_grad/DynamicStitchDynamicStitchgradients_5/Sum_6_grad/rangegradients_5/Sum_6_grad/modgradients_5/Sum_6_grad/Shapegradients_5/Sum_6_grad/Fill*
_output_shapes
:*/
_class%
#!loc:@gradients_5/Sum_6_grad/Shape*
N*
T0
?
 gradients_5/Sum_6_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :*/
_class%
#!loc:@gradients_5/Sum_6_grad/Shape
?
gradients_5/Sum_6_grad/MaximumMaximum$gradients_5/Sum_6_grad/DynamicStitch gradients_5/Sum_6_grad/Maximum/y*
_output_shapes
:*/
_class%
#!loc:@gradients_5/Sum_6_grad/Shape*
T0
?
gradients_5/Sum_6_grad/floordivFloorDivgradients_5/Sum_6_grad/Shapegradients_5/Sum_6_grad/Maximum*
_output_shapes
:*/
_class%
#!loc:@gradients_5/Sum_6_grad/Shape*
T0
?
gradients_5/Sum_6_grad/ReshapeReshapegradients_5/add_21_grad/Reshape$gradients_5/Sum_6_grad/DynamicStitch*
Tshape0*0
_output_shapes
:??????????????????*
T0
?
gradients_5/Sum_6_grad/TileTilegradients_5/Sum_6_grad/Reshapegradients_5/Sum_6_grad/floordiv*'
_output_shapes
:?????????*

Tmultiples0*
T0
a
gradients_5/Sum_7_grad/ShapeShapemul_4*
out_type0*
T0*
_output_shapes
:
?
gradients_5/Sum_7_grad/SizeConst*/
_class%
#!loc:@gradients_5/Sum_7_grad/Shape*
dtype0*
value	B :*
_output_shapes
: 
?
gradients_5/Sum_7_grad/addAddSum_7/reduction_indicesgradients_5/Sum_7_grad/Size*/
_class%
#!loc:@gradients_5/Sum_7_grad/Shape*
_output_shapes
: *
T0
?
gradients_5/Sum_7_grad/modFloorModgradients_5/Sum_7_grad/addgradients_5/Sum_7_grad/Size*
T0*
_output_shapes
: */
_class%
#!loc:@gradients_5/Sum_7_grad/Shape
?
gradients_5/Sum_7_grad/Shape_1Const*/
_class%
#!loc:@gradients_5/Sum_7_grad/Shape*
_output_shapes
: *
dtype0*
valueB 
?
"gradients_5/Sum_7_grad/range/startConst*
_output_shapes
: *
value	B : *
dtype0*/
_class%
#!loc:@gradients_5/Sum_7_grad/Shape
?
"gradients_5/Sum_7_grad/range/deltaConst*
dtype0*/
_class%
#!loc:@gradients_5/Sum_7_grad/Shape*
value	B :*
_output_shapes
: 
?
gradients_5/Sum_7_grad/rangeRange"gradients_5/Sum_7_grad/range/startgradients_5/Sum_7_grad/Size"gradients_5/Sum_7_grad/range/delta*
_output_shapes
:*/
_class%
#!loc:@gradients_5/Sum_7_grad/Shape*

Tidx0
?
!gradients_5/Sum_7_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*/
_class%
#!loc:@gradients_5/Sum_7_grad/Shape
?
gradients_5/Sum_7_grad/FillFillgradients_5/Sum_7_grad/Shape_1!gradients_5/Sum_7_grad/Fill/value*

index_type0*
_output_shapes
: */
_class%
#!loc:@gradients_5/Sum_7_grad/Shape*
T0
?
$gradients_5/Sum_7_grad/DynamicStitchDynamicStitchgradients_5/Sum_7_grad/rangegradients_5/Sum_7_grad/modgradients_5/Sum_7_grad/Shapegradients_5/Sum_7_grad/Fill*
N*
T0*/
_class%
#!loc:@gradients_5/Sum_7_grad/Shape*
_output_shapes
:
?
 gradients_5/Sum_7_grad/Maximum/yConst*/
_class%
#!loc:@gradients_5/Sum_7_grad/Shape*
dtype0*
_output_shapes
: *
value	B :
?
gradients_5/Sum_7_grad/MaximumMaximum$gradients_5/Sum_7_grad/DynamicStitch gradients_5/Sum_7_grad/Maximum/y*
T0*/
_class%
#!loc:@gradients_5/Sum_7_grad/Shape*
_output_shapes
:
?
gradients_5/Sum_7_grad/floordivFloorDivgradients_5/Sum_7_grad/Shapegradients_5/Sum_7_grad/Maximum*
T0*/
_class%
#!loc:@gradients_5/Sum_7_grad/Shape*
_output_shapes
:
?
gradients_5/Sum_7_grad/ReshapeReshape!gradients_5/add_21_grad/Reshape_1$gradients_5/Sum_7_grad/DynamicStitch*
Tshape0*
T0*0
_output_shapes
:??????????????????
?
gradients_5/Sum_7_grad/TileTilegradients_5/Sum_7_grad/Reshapegradients_5/Sum_7_grad/floordiv*
T0*

Tmultiples0*'
_output_shapes
:?????????
c
gradients_5/add_23_grad/ShapeShapeSum_12*
out_type0*
T0*
_output_shapes
:
d
gradients_5/add_23_grad/Shape_1Shapesub_8*
out_type0*
T0*
_output_shapes
:
?
-gradients_5/add_23_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/add_23_grad/Shapegradients_5/add_23_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_5/add_23_grad/SumSumgradients_5/add_24_grad/Reshape-gradients_5/add_23_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
gradients_5/add_23_grad/ReshapeReshapegradients_5/add_23_grad/Sumgradients_5/add_23_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients_5/add_23_grad/Sum_1Sumgradients_5/add_24_grad/Reshape/gradients_5/add_23_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
!gradients_5/add_23_grad/Reshape_1Reshapegradients_5/add_23_grad/Sum_1gradients_5/add_23_grad/Shape_1*'
_output_shapes
:?????????*
Tshape0*
T0
b
gradients_5/mul_3_grad/ShapeShapeNeg_11*
out_type0*
T0*
_output_shapes
:
c
gradients_5/mul_3_grad/Shape_1ShapeLog_9*
_output_shapes
:*
out_type0*
T0
?
,gradients_5/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/mul_3_grad/Shapegradients_5/mul_3_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
w
gradients_5/mul_3_grad/MulMulgradients_5/Sum_6_grad/TileLog_9*'
_output_shapes
:?????????*
T0
?
gradients_5/mul_3_grad/SumSumgradients_5/mul_3_grad/Mul,gradients_5/mul_3_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
gradients_5/mul_3_grad/ReshapeReshapegradients_5/mul_3_grad/Sumgradients_5/mul_3_grad/Shape*'
_output_shapes
:?????????*
Tshape0*
T0
z
gradients_5/mul_3_grad/Mul_1MulNeg_11gradients_5/Sum_6_grad/Tile*
T0*'
_output_shapes
:?????????
?
gradients_5/mul_3_grad/Sum_1Sumgradients_5/mul_3_grad/Mul_1.gradients_5/mul_3_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
?
 gradients_5/mul_3_grad/Reshape_1Reshapegradients_5/mul_3_grad/Sum_1gradients_5/mul_3_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
b
gradients_5/mul_4_grad/ShapeShapeNeg_12*
out_type0*
T0*
_output_shapes
:
d
gradients_5/mul_4_grad/Shape_1ShapeLog_10*
_output_shapes
:*
out_type0*
T0
?
,gradients_5/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/mul_4_grad/Shapegradients_5/mul_4_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
x
gradients_5/mul_4_grad/MulMulgradients_5/Sum_7_grad/TileLog_10*
T0*'
_output_shapes
:?????????
?
gradients_5/mul_4_grad/SumSumgradients_5/mul_4_grad/Mul,gradients_5/mul_4_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
gradients_5/mul_4_grad/ReshapeReshapegradients_5/mul_4_grad/Sumgradients_5/mul_4_grad/Shape*'
_output_shapes
:?????????*
Tshape0*
T0
z
gradients_5/mul_4_grad/Mul_1MulNeg_12gradients_5/Sum_7_grad/Tile*
T0*'
_output_shapes
:?????????
?
gradients_5/mul_4_grad/Sum_1Sumgradients_5/mul_4_grad/Mul_1.gradients_5/mul_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
?
 gradients_5/mul_4_grad/Reshape_1Reshapegradients_5/mul_4_grad/Sum_1gradients_5/mul_4_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:?????????
b
gradients_5/Sum_12_grad/ShapeShapemul_5*
out_type0*
_output_shapes
:*
T0
?
gradients_5/Sum_12_grad/SizeConst*
dtype0*
value	B :*0
_class&
$"loc:@gradients_5/Sum_12_grad/Shape*
_output_shapes
: 
?
gradients_5/Sum_12_grad/addAddSum_12/reduction_indicesgradients_5/Sum_12_grad/Size*
T0*
_output_shapes
: *0
_class&
$"loc:@gradients_5/Sum_12_grad/Shape
?
gradients_5/Sum_12_grad/modFloorModgradients_5/Sum_12_grad/addgradients_5/Sum_12_grad/Size*0
_class&
$"loc:@gradients_5/Sum_12_grad/Shape*
_output_shapes
: *
T0
?
gradients_5/Sum_12_grad/Shape_1Const*
_output_shapes
: *0
_class&
$"loc:@gradients_5/Sum_12_grad/Shape*
dtype0*
valueB 
?
#gradients_5/Sum_12_grad/range/startConst*0
_class&
$"loc:@gradients_5/Sum_12_grad/Shape*
dtype0*
_output_shapes
: *
value	B : 
?
#gradients_5/Sum_12_grad/range/deltaConst*
value	B :*0
_class&
$"loc:@gradients_5/Sum_12_grad/Shape*
_output_shapes
: *
dtype0
?
gradients_5/Sum_12_grad/rangeRange#gradients_5/Sum_12_grad/range/startgradients_5/Sum_12_grad/Size#gradients_5/Sum_12_grad/range/delta*0
_class&
$"loc:@gradients_5/Sum_12_grad/Shape*
_output_shapes
:*

Tidx0
?
"gradients_5/Sum_12_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*0
_class&
$"loc:@gradients_5/Sum_12_grad/Shape
?
gradients_5/Sum_12_grad/FillFillgradients_5/Sum_12_grad/Shape_1"gradients_5/Sum_12_grad/Fill/value*0
_class&
$"loc:@gradients_5/Sum_12_grad/Shape*

index_type0*
T0*
_output_shapes
: 
?
%gradients_5/Sum_12_grad/DynamicStitchDynamicStitchgradients_5/Sum_12_grad/rangegradients_5/Sum_12_grad/modgradients_5/Sum_12_grad/Shapegradients_5/Sum_12_grad/Fill*0
_class&
$"loc:@gradients_5/Sum_12_grad/Shape*
_output_shapes
:*
T0*
N
?
!gradients_5/Sum_12_grad/Maximum/yConst*0
_class&
$"loc:@gradients_5/Sum_12_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
gradients_5/Sum_12_grad/MaximumMaximum%gradients_5/Sum_12_grad/DynamicStitch!gradients_5/Sum_12_grad/Maximum/y*
T0*0
_class&
$"loc:@gradients_5/Sum_12_grad/Shape*
_output_shapes
:
?
 gradients_5/Sum_12_grad/floordivFloorDivgradients_5/Sum_12_grad/Shapegradients_5/Sum_12_grad/Maximum*
T0*
_output_shapes
:*0
_class&
$"loc:@gradients_5/Sum_12_grad/Shape
?
gradients_5/Sum_12_grad/ReshapeReshapegradients_5/add_23_grad/Reshape%gradients_5/Sum_12_grad/DynamicStitch*0
_output_shapes
:??????????????????*
Tshape0*
T0
?
gradients_5/Sum_12_grad/TileTilegradients_5/Sum_12_grad/Reshape gradients_5/Sum_12_grad/floordiv*
T0*'
_output_shapes
:?????????*

Tmultiples0
b
gradients_5/sub_8_grad/ShapeShapeLgamma*
T0*
out_type0*
_output_shapes
:
d
gradients_5/sub_8_grad/Shape_1ShapeSum_10*
T0*
_output_shapes
:*
out_type0
?
,gradients_5/sub_8_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/sub_8_grad/Shapegradients_5/sub_8_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_5/sub_8_grad/SumSum!gradients_5/add_23_grad/Reshape_1,gradients_5/sub_8_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
gradients_5/sub_8_grad/ReshapeReshapegradients_5/sub_8_grad/Sumgradients_5/sub_8_grad/Shape*
T0*'
_output_shapes
:?????????*
Tshape0
?
gradients_5/sub_8_grad/Sum_1Sum!gradients_5/add_23_grad/Reshape_1.gradients_5/sub_8_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
b
gradients_5/sub_8_grad/NegNeggradients_5/sub_8_grad/Sum_1*
_output_shapes
:*
T0
?
 gradients_5/sub_8_grad/Reshape_1Reshapegradients_5/sub_8_grad/Neggradients_5/sub_8_grad/Shape_1*
Tshape0*'
_output_shapes
:?????????*
T0
?
!gradients_5/Log_9_grad/Reciprocal
Reciprocaladd_19!^gradients_5/mul_3_grad/Reshape_1*
T0*'
_output_shapes
:?????????
?
gradients_5/Log_9_grad/mulMul gradients_5/mul_3_grad/Reshape_1!gradients_5/Log_9_grad/Reciprocal*'
_output_shapes
:?????????*
T0
?
"gradients_5/Log_10_grad/Reciprocal
Reciprocaladd_20!^gradients_5/mul_4_grad/Reshape_1*'
_output_shapes
:?????????*
T0
?
gradients_5/Log_10_grad/mulMul gradients_5/mul_4_grad/Reshape_1"gradients_5/Log_10_grad/Reciprocal*'
_output_shapes
:?????????*
T0
b
gradients_5/mul_5_grad/ShapeShapesub_10*
out_type0*
T0*
_output_shapes
:
d
gradients_5/mul_5_grad/Shape_1Shapesub_11*
out_type0*
T0*
_output_shapes
:
?
,gradients_5/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/mul_5_grad/Shapegradients_5/mul_5_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
y
gradients_5/mul_5_grad/MulMulgradients_5/Sum_12_grad/Tilesub_11*
T0*'
_output_shapes
:?????????
?
gradients_5/mul_5_grad/SumSumgradients_5/mul_5_grad/Mul,gradients_5/mul_5_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
?
gradients_5/mul_5_grad/ReshapeReshapegradients_5/mul_5_grad/Sumgradients_5/mul_5_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
{
gradients_5/mul_5_grad/Mul_1Mulsub_10gradients_5/Sum_12_grad/Tile*
T0*'
_output_shapes
:?????????
?
gradients_5/mul_5_grad/Sum_1Sumgradients_5/mul_5_grad/Mul_1.gradients_5/mul_5_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
 gradients_5/mul_5_grad/Reshape_1Reshapegradients_5/mul_5_grad/Sum_1gradients_5/mul_5_grad/Shape_1*
T0*'
_output_shapes
:?????????*
Tshape0
?
gradients_5/Lgamma_grad/DigammaDigammaSum_8^gradients_5/sub_8_grad/Reshape*
T0*'
_output_shapes
:?????????
?
gradients_5/Lgamma_grad/mulMulgradients_5/sub_8_grad/Reshapegradients_5/Lgamma_grad/Digamma*'
_output_shapes
:?????????*
T0
e
gradients_5/Sum_10_grad/ShapeShapeLgamma_1*
_output_shapes
:*
T0*
out_type0
?
gradients_5/Sum_10_grad/SizeConst*0
_class&
$"loc:@gradients_5/Sum_10_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
gradients_5/Sum_10_grad/addAddSum_10/reduction_indicesgradients_5/Sum_10_grad/Size*
_output_shapes
: *0
_class&
$"loc:@gradients_5/Sum_10_grad/Shape*
T0
?
gradients_5/Sum_10_grad/modFloorModgradients_5/Sum_10_grad/addgradients_5/Sum_10_grad/Size*
T0*0
_class&
$"loc:@gradients_5/Sum_10_grad/Shape*
_output_shapes
: 
?
gradients_5/Sum_10_grad/Shape_1Const*
dtype0*0
_class&
$"loc:@gradients_5/Sum_10_grad/Shape*
_output_shapes
: *
valueB 
?
#gradients_5/Sum_10_grad/range/startConst*0
_class&
$"loc:@gradients_5/Sum_10_grad/Shape*
_output_shapes
: *
dtype0*
value	B : 
?
#gradients_5/Sum_10_grad/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0*0
_class&
$"loc:@gradients_5/Sum_10_grad/Shape
?
gradients_5/Sum_10_grad/rangeRange#gradients_5/Sum_10_grad/range/startgradients_5/Sum_10_grad/Size#gradients_5/Sum_10_grad/range/delta*0
_class&
$"loc:@gradients_5/Sum_10_grad/Shape*
_output_shapes
:*

Tidx0
?
"gradients_5/Sum_10_grad/Fill/valueConst*
value	B :*
dtype0*0
_class&
$"loc:@gradients_5/Sum_10_grad/Shape*
_output_shapes
: 
?
gradients_5/Sum_10_grad/FillFillgradients_5/Sum_10_grad/Shape_1"gradients_5/Sum_10_grad/Fill/value*
T0*

index_type0*0
_class&
$"loc:@gradients_5/Sum_10_grad/Shape*
_output_shapes
: 
?
%gradients_5/Sum_10_grad/DynamicStitchDynamicStitchgradients_5/Sum_10_grad/rangegradients_5/Sum_10_grad/modgradients_5/Sum_10_grad/Shapegradients_5/Sum_10_grad/Fill*0
_class&
$"loc:@gradients_5/Sum_10_grad/Shape*
N*
_output_shapes
:*
T0
?
!gradients_5/Sum_10_grad/Maximum/yConst*0
_class&
$"loc:@gradients_5/Sum_10_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
gradients_5/Sum_10_grad/MaximumMaximum%gradients_5/Sum_10_grad/DynamicStitch!gradients_5/Sum_10_grad/Maximum/y*
T0*
_output_shapes
:*0
_class&
$"loc:@gradients_5/Sum_10_grad/Shape
?
 gradients_5/Sum_10_grad/floordivFloorDivgradients_5/Sum_10_grad/Shapegradients_5/Sum_10_grad/Maximum*0
_class&
$"loc:@gradients_5/Sum_10_grad/Shape*
_output_shapes
:*
T0
?
gradients_5/Sum_10_grad/ReshapeReshape gradients_5/sub_8_grad/Reshape_1%gradients_5/Sum_10_grad/DynamicStitch*
T0*0
_output_shapes
:??????????????????*
Tshape0
?
gradients_5/Sum_10_grad/TileTilegradients_5/Sum_10_grad/Reshape gradients_5/Sum_10_grad/floordiv*'
_output_shapes
:?????????*
T0*

Tmultiples0
f
gradients_5/add_19_grad/ShapeShape	Sigmoid_4*
_output_shapes
:*
T0*
out_type0
b
gradients_5/add_19_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
?
-gradients_5/add_19_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/add_19_grad/Shapegradients_5/add_19_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_5/add_19_grad/SumSumgradients_5/Log_9_grad/mul-gradients_5/add_19_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
gradients_5/add_19_grad/ReshapeReshapegradients_5/add_19_grad/Sumgradients_5/add_19_grad/Shape*'
_output_shapes
:?????????*
Tshape0*
T0
?
gradients_5/add_19_grad/Sum_1Sumgradients_5/Log_9_grad/mul/gradients_5/add_19_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
?
!gradients_5/add_19_grad/Reshape_1Reshapegradients_5/add_19_grad/Sum_1gradients_5/add_19_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
b
gradients_5/add_20_grad/ShapeShapesub_7*
T0*
out_type0*
_output_shapes
:
b
gradients_5/add_20_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
?
-gradients_5/add_20_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/add_20_grad/Shapegradients_5/add_20_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_5/add_20_grad/SumSumgradients_5/Log_10_grad/mul-gradients_5/add_20_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
?
gradients_5/add_20_grad/ReshapeReshapegradients_5/add_20_grad/Sumgradients_5/add_20_grad/Shape*'
_output_shapes
:?????????*
T0*
Tshape0
?
gradients_5/add_20_grad/Sum_1Sumgradients_5/Log_10_grad/mul/gradients_5/add_20_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
?
!gradients_5/add_20_grad/Reshape_1Reshapegradients_5/add_20_grad/Sum_1gradients_5/add_20_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
d
gradients_5/sub_10_grad/ShapeShapeReshape*
_output_shapes
:*
out_type0*
T0
p
gradients_5/sub_10_grad/Shape_1Const*
dtype0*
valueB"      *
_output_shapes
:
?
-gradients_5/sub_10_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/sub_10_grad/Shapegradients_5/sub_10_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_5/sub_10_grad/SumSumgradients_5/mul_5_grad/Reshape-gradients_5/sub_10_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
?
gradients_5/sub_10_grad/ReshapeReshapegradients_5/sub_10_grad/Sumgradients_5/sub_10_grad/Shape*'
_output_shapes
:?????????*
Tshape0*
T0
?
gradients_5/sub_10_grad/Sum_1Sumgradients_5/mul_5_grad/Reshape/gradients_5/sub_10_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
d
gradients_5/sub_10_grad/NegNeggradients_5/sub_10_grad/Sum_1*
_output_shapes
:*
T0
?
!gradients_5/sub_10_grad/Reshape_1Reshapegradients_5/sub_10_grad/Neggradients_5/sub_10_grad/Shape_1*
Tshape0*
T0*
_output_shapes

:
f
gradients_5/sub_11_grad/ShapeShape	Digamma_1*
_output_shapes
:*
out_type0*
T0
f
gradients_5/sub_11_grad/Shape_1ShapeDigamma*
out_type0*
T0*
_output_shapes
:
?
-gradients_5/sub_11_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/sub_11_grad/Shapegradients_5/sub_11_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_5/sub_11_grad/SumSum gradients_5/mul_5_grad/Reshape_1-gradients_5/sub_11_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
?
gradients_5/sub_11_grad/ReshapeReshapegradients_5/sub_11_grad/Sumgradients_5/sub_11_grad/Shape*
Tshape0*'
_output_shapes
:?????????*
T0
?
gradients_5/sub_11_grad/Sum_1Sum gradients_5/mul_5_grad/Reshape_1/gradients_5/sub_11_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
d
gradients_5/sub_11_grad/NegNeggradients_5/sub_11_grad/Sum_1*
_output_shapes
:*
T0
?
!gradients_5/sub_11_grad/Reshape_1Reshapegradients_5/sub_11_grad/Neggradients_5/sub_11_grad/Shape_1*
T0*'
_output_shapes
:?????????*
Tshape0
?
!gradients_5/Lgamma_1_grad/DigammaDigammaReshape^gradients_5/Sum_10_grad/Tile*
T0*'
_output_shapes
:?????????
?
gradients_5/Lgamma_1_grad/mulMulgradients_5/Sum_10_grad/Tile!gradients_5/Lgamma_1_grad/Digamma*
T0*'
_output_shapes
:?????????
?
&gradients_5/Sigmoid_4_grad/SigmoidGradSigmoidGrad	Sigmoid_4gradients_5/add_19_grad/Reshape*
T0*'
_output_shapes
:?????????
_
gradients_5/sub_7_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
g
gradients_5/sub_7_grad/Shape_1Shape	Sigmoid_5*
out_type0*
T0*
_output_shapes
:
?
,gradients_5/sub_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/sub_7_grad/Shapegradients_5/sub_7_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_5/sub_7_grad/SumSumgradients_5/add_20_grad/Reshape,gradients_5/sub_7_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
gradients_5/sub_7_grad/ReshapeReshapegradients_5/sub_7_grad/Sumgradients_5/sub_7_grad/Shape*
Tshape0*
_output_shapes
: *
T0
?
gradients_5/sub_7_grad/Sum_1Sumgradients_5/add_20_grad/Reshape.gradients_5/sub_7_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
b
gradients_5/sub_7_grad/NegNeggradients_5/sub_7_grad/Sum_1*
_output_shapes
:*
T0
?
 gradients_5/sub_7_grad/Reshape_1Reshapegradients_5/sub_7_grad/Neggradients_5/sub_7_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
 gradients_5/Digamma_1_grad/ConstConst ^gradients_5/sub_11_grad/Reshape*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
$gradients_5/Digamma_1_grad/Polygamma	Polygamma gradients_5/Digamma_1_grad/ConstReshape*
T0*'
_output_shapes
:?????????
?
gradients_5/Digamma_1_grad/mulMulgradients_5/sub_11_grad/Reshape$gradients_5/Digamma_1_grad/Polygamma*'
_output_shapes
:?????????*
T0
?
gradients_5/Digamma_grad/ConstConst"^gradients_5/sub_11_grad/Reshape_1*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
"gradients_5/Digamma_grad/Polygamma	Polygammagradients_5/Digamma_grad/ConstSum_8*'
_output_shapes
:?????????*
T0
?
gradients_5/Digamma_grad/mulMul!gradients_5/sub_11_grad/Reshape_1"gradients_5/Digamma_grad/Polygamma*
T0*'
_output_shapes
:?????????
?
&gradients_5/Sigmoid_5_grad/SigmoidGradSigmoidGrad	Sigmoid_5 gradients_5/sub_7_grad/Reshape_1*'
_output_shapes
:?????????*
T0
?
gradients_5/AddNAddNgradients_5/Lgamma_grad/mulgradients_5/Digamma_grad/mul*.
_class$
" loc:@gradients_5/Lgamma_grad/mul*'
_output_shapes
:?????????*
T0*
N
c
gradients_5/Sum_8_grad/ShapeShapeReshape*
out_type0*
T0*
_output_shapes
:
?
gradients_5/Sum_8_grad/SizeConst*
dtype0*/
_class%
#!loc:@gradients_5/Sum_8_grad/Shape*
value	B :*
_output_shapes
: 
?
gradients_5/Sum_8_grad/addAddSum_8/reduction_indicesgradients_5/Sum_8_grad/Size*
T0*/
_class%
#!loc:@gradients_5/Sum_8_grad/Shape*
_output_shapes
: 
?
gradients_5/Sum_8_grad/modFloorModgradients_5/Sum_8_grad/addgradients_5/Sum_8_grad/Size*
_output_shapes
: */
_class%
#!loc:@gradients_5/Sum_8_grad/Shape*
T0
?
gradients_5/Sum_8_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB */
_class%
#!loc:@gradients_5/Sum_8_grad/Shape
?
"gradients_5/Sum_8_grad/range/startConst*
_output_shapes
: *
value	B : */
_class%
#!loc:@gradients_5/Sum_8_grad/Shape*
dtype0
?
"gradients_5/Sum_8_grad/range/deltaConst*/
_class%
#!loc:@gradients_5/Sum_8_grad/Shape*
value	B :*
_output_shapes
: *
dtype0
?
gradients_5/Sum_8_grad/rangeRange"gradients_5/Sum_8_grad/range/startgradients_5/Sum_8_grad/Size"gradients_5/Sum_8_grad/range/delta*/
_class%
#!loc:@gradients_5/Sum_8_grad/Shape*

Tidx0*
_output_shapes
:
?
!gradients_5/Sum_8_grad/Fill/valueConst*/
_class%
#!loc:@gradients_5/Sum_8_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
gradients_5/Sum_8_grad/FillFillgradients_5/Sum_8_grad/Shape_1!gradients_5/Sum_8_grad/Fill/value*
_output_shapes
: *

index_type0*/
_class%
#!loc:@gradients_5/Sum_8_grad/Shape*
T0
?
$gradients_5/Sum_8_grad/DynamicStitchDynamicStitchgradients_5/Sum_8_grad/rangegradients_5/Sum_8_grad/modgradients_5/Sum_8_grad/Shapegradients_5/Sum_8_grad/Fill*
N*/
_class%
#!loc:@gradients_5/Sum_8_grad/Shape*
T0*
_output_shapes
:
?
 gradients_5/Sum_8_grad/Maximum/yConst*
_output_shapes
: *
dtype0*/
_class%
#!loc:@gradients_5/Sum_8_grad/Shape*
value	B :
?
gradients_5/Sum_8_grad/MaximumMaximum$gradients_5/Sum_8_grad/DynamicStitch gradients_5/Sum_8_grad/Maximum/y*
_output_shapes
:*
T0*/
_class%
#!loc:@gradients_5/Sum_8_grad/Shape
?
gradients_5/Sum_8_grad/floordivFloorDivgradients_5/Sum_8_grad/Shapegradients_5/Sum_8_grad/Maximum*
_output_shapes
:*
T0*/
_class%
#!loc:@gradients_5/Sum_8_grad/Shape
?
gradients_5/Sum_8_grad/ReshapeReshapegradients_5/AddN$gradients_5/Sum_8_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
gradients_5/Sum_8_grad/TileTilegradients_5/Sum_8_grad/Reshapegradients_5/Sum_8_grad/floordiv*
T0*

Tmultiples0*'
_output_shapes
:?????????
?
3gradients_5/disc_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad&gradients_5/Sigmoid_5_grad/SigmoidGrad*
data_formatNHWC*
T0*
_output_shapes
:
?
gradients_5/AddN_1AddNgradients_5/sub_10_grad/Reshapegradients_5/Lgamma_1_grad/mulgradients_5/Digamma_1_grad/mulgradients_5/Sum_8_grad/Tile*
N*'
_output_shapes
:?????????*
T0*2
_class(
&$loc:@gradients_5/sub_10_grad/Reshape
f
gradients_5/Reshape_grad/ShapeShapeGatherNd*
T0*
out_type0*
_output_shapes
:
?
 gradients_5/Reshape_grad/ReshapeReshapegradients_5/AddN_1gradients_5/Reshape_grad/Shape*#
_output_shapes
:?????????*
Tshape0*
T0
?
-gradients_5/disc_1/dense_1/MatMul_grad/MatMulMatMul&gradients_5/Sigmoid_5_grad/SigmoidGraddisc/dense_1/kernel/read*
T0*
transpose_a( *
transpose_b(*(
_output_shapes
:??????????
?
/gradients_5/disc_1/dense_1/MatMul_grad/MatMul_1MatMuldisc_1/dense/Relu&gradients_5/Sigmoid_5_grad/SigmoidGrad*
T0*
transpose_a(*
transpose_b( *
_output_shapes
:	?
e
gradients_5/GatherNd_grad/ShapeShapeadd_22*
T0*
out_type0	*
_output_shapes
:
?
#gradients_5/GatherNd_grad/ScatterNd	ScatterNdWhere gradients_5/Reshape_grad/Reshapegradients_5/GatherNd_grad/Shape*'
_output_shapes
:?????????*
T0*
Tindices0	
?
+gradients_5/disc_1/dense/Relu_grad/ReluGradReluGrad-gradients_5/disc_1/dense_1/MatMul_grad/MatMuldisc_1/dense/Relu*(
_output_shapes
:??????????*
T0
`
gradients_5/add_22_grad/ShapeShapeExp*
T0*
_output_shapes
:*
out_type0
b
gradients_5/add_22_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
?
-gradients_5/add_22_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/add_22_grad/Shapegradients_5/add_22_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_5/add_22_grad/SumSum#gradients_5/GatherNd_grad/ScatterNd-gradients_5/add_22_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
gradients_5/add_22_grad/ReshapeReshapegradients_5/add_22_grad/Sumgradients_5/add_22_grad/Shape*
Tshape0*'
_output_shapes
:?????????*
T0
?
gradients_5/add_22_grad/Sum_1Sum#gradients_5/GatherNd_grad/ScatterNd/gradients_5/add_22_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
?
!gradients_5/add_22_grad/Reshape_1Reshapegradients_5/add_22_grad/Sum_1gradients_5/add_22_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
1gradients_5/disc_1/dense/BiasAdd_grad/BiasAddGradBiasAddGrad+gradients_5/disc_1/dense/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:?*
T0
w
gradients_5/Exp_grad/mulMulgradients_5/add_22_grad/ReshapeExp*'
_output_shapes
:?????????*
T0
?
+gradients_5/disc_1/dense/MatMul_grad/MatMulMatMul+gradients_5/disc_1/dense/Relu_grad/ReluGraddisc/dense/kernel/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:??????????
?
-gradients_5/disc_1/dense/MatMul_grad/MatMul_1MatMuldisc_1/flatten/Reshape+gradients_5/disc_1/dense/Relu_grad/ReluGrad*
transpose_a(* 
_output_shapes
:
??*
T0*
transpose_b( 
?
gradients_5/AddN_2AddN&gradients_5/Sigmoid_4_grad/SigmoidGradgradients_5/Exp_grad/mul*9
_class/
-+loc:@gradients_5/Sigmoid_4_grad/SigmoidGrad*
N*
T0*'
_output_shapes
:?????????
?
1gradients_5/disc/dense_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_5/AddN_2*
_output_shapes
:*
T0*
data_formatNHWC
}
-gradients_5/disc_1/flatten/Reshape_grad/ShapeShapedisc_1/MaxPool_1*
out_type0*
_output_shapes
:*
T0
?
/gradients_5/disc_1/flatten/Reshape_grad/ReshapeReshape+gradients_5/disc_1/dense/MatMul_grad/MatMul-gradients_5/disc_1/flatten/Reshape_grad/Shape*
Tshape0*/
_output_shapes
:?????????2*
T0
?
+gradients_5/disc/dense_1/MatMul_grad/MatMulMatMulgradients_5/AddN_2disc/dense_1/kernel/read*
transpose_b(*
T0*(
_output_shapes
:??????????*
transpose_a( 
?
-gradients_5/disc/dense_1/MatMul_grad/MatMul_1MatMuldisc/dense/Relugradients_5/AddN_2*
transpose_a(*
T0*
transpose_b( *
_output_shapes
:	?
?
-gradients_5/disc_1/MaxPool_1_grad/MaxPoolGradMaxPoolGraddisc_1/conv2d_1/Reludisc_1/MaxPool_1/gradients_5/disc_1/flatten/Reshape_grad/Reshape*
T0*
data_formatNHWC*
ksize
*/
_output_shapes
:?????????2*
paddingSAME*
strides

?
)gradients_5/disc/dense/Relu_grad/ReluGradReluGrad+gradients_5/disc/dense_1/MatMul_grad/MatMuldisc/dense/Relu*(
_output_shapes
:??????????*
T0
?
.gradients_5/disc_1/conv2d_1/Relu_grad/ReluGradReluGrad-gradients_5/disc_1/MaxPool_1_grad/MaxPoolGraddisc_1/conv2d_1/Relu*/
_output_shapes
:?????????2*
T0
?
/gradients_5/disc/dense/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_5/disc/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:?
?
4gradients_5/disc_1/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients_5/disc_1/conv2d_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:2
?
)gradients_5/disc/dense/MatMul_grad/MatMulMatMul)gradients_5/disc/dense/Relu_grad/ReluGraddisc/dense/kernel/read*
T0*(
_output_shapes
:??????????*
transpose_b(*
transpose_a( 
?
+gradients_5/disc/dense/MatMul_grad/MatMul_1MatMuldisc/flatten/Reshape)gradients_5/disc/dense/Relu_grad/ReluGrad*
T0* 
_output_shapes
:
??*
transpose_b( *
transpose_a(
?
.gradients_5/disc_1/conv2d_1/Conv2D_grad/ShapeNShapeNdisc_1/MaxPooldisc/conv2d_1/kernel/read*
T0* 
_output_shapes
::*
out_type0*
N
?
;gradients_5/disc_1/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput.gradients_5/disc_1/conv2d_1/Conv2D_grad/ShapeNdisc/conv2d_1/kernel/read.gradients_5/disc_1/conv2d_1/Relu_grad/ReluGrad*
paddingVALID*
data_formatNHWC*/
_output_shapes
:?????????*
T0*
	dilations
*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
?
<gradients_5/disc_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdisc_1/MaxPool0gradients_5/disc_1/conv2d_1/Conv2D_grad/ShapeN:1.gradients_5/disc_1/conv2d_1/Relu_grad/ReluGrad*
	dilations
*&
_output_shapes
:2*
T0*
explicit_paddings
 *
use_cudnn_on_gpu(*
strides
*
paddingVALID*
data_formatNHWC
y
+gradients_5/disc/flatten/Reshape_grad/ShapeShapedisc/MaxPool_1*
_output_shapes
:*
out_type0*
T0
?
-gradients_5/disc/flatten/Reshape_grad/ReshapeReshape)gradients_5/disc/dense/MatMul_grad/MatMul+gradients_5/disc/flatten/Reshape_grad/Shape*
T0*/
_output_shapes
:?????????2*
Tshape0
?
+gradients_5/disc_1/MaxPool_grad/MaxPoolGradMaxPoolGraddisc_1/conv2d/Reludisc_1/MaxPool;gradients_5/disc_1/conv2d_1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????*
strides
*
paddingSAME*
T0*
data_formatNHWC*
ksize

?
+gradients_5/disc/MaxPool_1_grad/MaxPoolGradMaxPoolGraddisc/conv2d_1/Reludisc/MaxPool_1-gradients_5/disc/flatten/Reshape_grad/Reshape*
ksize
*
strides
*
data_formatNHWC*/
_output_shapes
:?????????2*
paddingSAME*
T0
?
,gradients_5/disc_1/conv2d/Relu_grad/ReluGradReluGrad+gradients_5/disc_1/MaxPool_grad/MaxPoolGraddisc_1/conv2d/Relu*
T0*/
_output_shapes
:?????????
?
,gradients_5/disc/conv2d_1/Relu_grad/ReluGradReluGrad+gradients_5/disc/MaxPool_1_grad/MaxPoolGraddisc/conv2d_1/Relu*/
_output_shapes
:?????????2*
T0
?
2gradients_5/disc_1/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_5/disc_1/conv2d/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:
?
2gradients_5/disc/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_5/disc/conv2d_1/Relu_grad/ReluGrad*
T0*
_output_shapes
:2*
data_formatNHWC
?
,gradients_5/disc_1/conv2d/Conv2D_grad/ShapeNShapeNdisc_1/Reshapedisc/conv2d/kernel/read*
N*
T0*
out_type0* 
_output_shapes
::
?
9gradients_5/disc_1/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput,gradients_5/disc_1/conv2d/Conv2D_grad/ShapeNdisc/conv2d/kernel/read,gradients_5/disc_1/conv2d/Relu_grad/ReluGrad*
strides
*
	dilations
*
paddingVALID*
explicit_paddings
 *
use_cudnn_on_gpu(*
data_formatNHWC*
T0*/
_output_shapes
:?????????
?
:gradients_5/disc_1/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdisc_1/Reshape.gradients_5/disc_1/conv2d/Conv2D_grad/ShapeN:1,gradients_5/disc_1/conv2d/Relu_grad/ReluGrad*
data_formatNHWC*
	dilations
*
use_cudnn_on_gpu(*
paddingVALID*
T0*
strides
*
explicit_paddings
 *&
_output_shapes
:
?
,gradients_5/disc/conv2d_1/Conv2D_grad/ShapeNShapeNdisc/MaxPooldisc/conv2d_1/kernel/read*
N*
out_type0* 
_output_shapes
::*
T0
?
9gradients_5/disc/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput,gradients_5/disc/conv2d_1/Conv2D_grad/ShapeNdisc/conv2d_1/kernel/read,gradients_5/disc/conv2d_1/Relu_grad/ReluGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 */
_output_shapes
:?????????*
paddingVALID*
T0*
	dilations

?
:gradients_5/disc/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdisc/MaxPool.gradients_5/disc/conv2d_1/Conv2D_grad/ShapeN:1,gradients_5/disc/conv2d_1/Relu_grad/ReluGrad*
strides
*
paddingVALID*
	dilations
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *&
_output_shapes
:2*
T0

%gradients_5/disc_1/Reshape_grad/ShapeShapedecoder_1/generated_images*
_output_shapes
:*
out_type0*
T0
?
'gradients_5/disc_1/Reshape_grad/ReshapeReshape9gradients_5/disc_1/conv2d/Conv2D_grad/Conv2DBackpropInput%gradients_5/disc_1/Reshape_grad/Shape*/
_output_shapes
:?????????*
Tshape0*
T0
?
)gradients_5/disc/MaxPool_grad/MaxPoolGradMaxPoolGraddisc/conv2d/Reludisc/MaxPool9gradients_5/disc/conv2d_1/Conv2D_grad/Conv2DBackpropInput*
ksize
*
data_formatNHWC*
strides
*/
_output_shapes
:?????????*
paddingSAME*
T0
?
*gradients_5/disc/conv2d/Relu_grad/ReluGradReluGrad)gradients_5/disc/MaxPool_grad/MaxPoolGraddisc/conv2d/Relu*
T0*/
_output_shapes
:?????????
?
0gradients_5/decoder_1/layer_2/Tanh_grad/TanhGradTanhGraddecoder_1/layer_2/Tanh'gradients_5/disc_1/Reshape_grad/Reshape*/
_output_shapes
:?????????*
T0
?
0gradients_5/disc/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients_5/disc/conv2d/Relu_grad/ReluGrad*
_output_shapes
:*
data_formatNHWC*
T0
?
6gradients_5/decoder_1/layer_2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients_5/decoder_1/layer_2/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes
:*
T0
?
*gradients_5/disc/conv2d/Conv2D_grad/ShapeNShapeNdisc/Reshapedisc/conv2d/kernel/read* 
_output_shapes
::*
out_type0*
N*
T0
?
7gradients_5/disc/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients_5/disc/conv2d/Conv2D_grad/ShapeNdisc/conv2d/kernel/read*gradients_5/disc/conv2d/Relu_grad/ReluGrad*
explicit_paddings
 *
strides
*
	dilations
*/
_output_shapes
:?????????*
use_cudnn_on_gpu(*
paddingVALID*
T0*
data_formatNHWC
?
8gradients_5/disc/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdisc/Reshape,gradients_5/disc/conv2d/Conv2D_grad/ShapeN:1*gradients_5/disc/conv2d/Relu_grad/ReluGrad*
strides
*
T0*
use_cudnn_on_gpu(*&
_output_shapes
:*
paddingVALID*
explicit_paddings
 *
	dilations
*
data_formatNHWC
?
9gradients_5/decoder_1/layer_2/conv2d_transpose_grad/ShapeConst*%
valueB"         ?   *
dtype0*
_output_shapes
:
?
Hgradients_5/decoder_1/layer_2/conv2d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilter0gradients_5/decoder_1/layer_2/Tanh_grad/TanhGrad9gradients_5/decoder_1/layer_2/conv2d_transpose_grad/Shapedecoder_1/LeakyRelu_1*
paddingSAME*
	dilations
*
explicit_paddings
 *
use_cudnn_on_gpu(*
strides
*'
_output_shapes
:?*
T0*
data_formatNHWC
?
:gradients_5/decoder_1/layer_2/conv2d_transpose_grad/Conv2DConv2D0gradients_5/decoder_1/layer_2/Tanh_grad/TanhGraddecoder/layer_2/kernel/read*
	dilations
*
paddingSAME*
use_cudnn_on_gpu(*
explicit_paddings
 *
strides
*
data_formatNHWC*
T0*0
_output_shapes
:??????????
d
#gradients_5/disc/Reshape_grad/ShapeShapeX*
T0*
_output_shapes
:*
out_type0
?
%gradients_5/disc/Reshape_grad/ReshapeReshape7gradients_5/disc/conv2d/Conv2D_grad/Conv2DBackpropInput#gradients_5/disc/Reshape_grad/Shape*
Tshape0*
T0*(
_output_shapes
:??????????
?
4gradients_5/decoder_1/LeakyRelu_1_grad/LeakyReluGradLeakyReluGrad:gradients_5/decoder_1/layer_2/conv2d_transpose_grad/Conv2D.decoder_1/batch_normalization_1/FusedBatchNorm*0
_output_shapes
:??????????*
alpha%??L>*
T0
{
gradients_5/zeros_like	ZerosLike0decoder_1/batch_normalization_1/FusedBatchNorm:1*
T0*
_output_shapes	
:?
}
gradients_5/zeros_like_1	ZerosLike0decoder_1/batch_normalization_1/FusedBatchNorm:2*
_output_shapes	
:?*
T0
}
gradients_5/zeros_like_2	ZerosLike0decoder_1/batch_normalization_1/FusedBatchNorm:3*
T0*
_output_shapes	
:?
}
gradients_5/zeros_like_3	ZerosLike0decoder_1/batch_normalization_1/FusedBatchNorm:4*
T0*
_output_shapes	
:?
?
Rgradients_5/decoder_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGrad4gradients_5/decoder_1/LeakyRelu_1_grad/LeakyReluGraddecoder_1/layer_1/BiasAdd(decoder/batch_normalization_1/gamma/read0decoder_1/batch_normalization_1/FusedBatchNorm:30decoder_1/batch_normalization_1/FusedBatchNorm:4*
T0*
epsilon%o?:*F
_output_shapes4
2:??????????:?:?: : *
data_formatNHWC*
is_training(
?
6gradients_5/decoder_1/layer_1/BiasAdd_grad/BiasAddGradBiasAddGradRgradients_5/decoder_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*
data_formatNHWC*
_output_shapes	
:?*
T0
?
9gradients_5/decoder_1/layer_1/conv2d_transpose_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?      
?
Hgradients_5/decoder_1/layer_1/conv2d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilterRgradients_5/decoder_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad9gradients_5/decoder_1/layer_1/conv2d_transpose_grad/Shapedecoder_1/LeakyRelu*
paddingSAME*(
_output_shapes
:??*
explicit_paddings
 *
data_formatNHWC*
strides
*
	dilations
*
use_cudnn_on_gpu(*
T0
?
:gradients_5/decoder_1/layer_1/conv2d_transpose_grad/Conv2DConv2DRgradients_5/decoder_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGraddecoder/layer_1/kernel/read*
paddingSAME*
explicit_paddings
 *
strides
*
	dilations
*0
_output_shapes
:??????????*
use_cudnn_on_gpu(*
data_formatNHWC*
T0
?
2gradients_5/decoder_1/LeakyRelu_grad/LeakyReluGradLeakyReluGrad:gradients_5/decoder_1/layer_1/conv2d_transpose_grad/Conv2D,decoder_1/batch_normalization/FusedBatchNorm*
T0*
alpha%??L>*0
_output_shapes
:??????????
{
gradients_5/zeros_like_4	ZerosLike.decoder_1/batch_normalization/FusedBatchNorm:1*
_output_shapes	
:?*
T0
{
gradients_5/zeros_like_5	ZerosLike.decoder_1/batch_normalization/FusedBatchNorm:2*
T0*
_output_shapes	
:?
{
gradients_5/zeros_like_6	ZerosLike.decoder_1/batch_normalization/FusedBatchNorm:3*
_output_shapes	
:?*
T0
{
gradients_5/zeros_like_7	ZerosLike.decoder_1/batch_normalization/FusedBatchNorm:4*
_output_shapes	
:?*
T0
?
Pgradients_5/decoder_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGrad2gradients_5/decoder_1/LeakyRelu_grad/LeakyReluGraddecoder_1/layer_0/BiasAdd&decoder/batch_normalization/gamma/read.decoder_1/batch_normalization/FusedBatchNorm:3.decoder_1/batch_normalization/FusedBatchNorm:4*
epsilon%o?:*F
_output_shapes4
2:??????????:?:?: : *
T0*
data_formatNHWC*
is_training(
?
6gradients_5/decoder_1/layer_0/BiasAdd_grad/BiasAddGradBiasAddGradPgradients_5/decoder_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
data_formatNHWC*
_output_shapes	
:?
?
9gradients_5/decoder_1/layer_0/conv2d_transpose_grad/ShapeConst*%
valueB"         d   *
_output_shapes
:*
dtype0
?
Hgradients_5/decoder_1/layer_0/conv2d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilterPgradients_5/decoder_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad9gradients_5/decoder_1/layer_0/conv2d_transpose_grad/Shapedecoder_1/Reshape*'
_output_shapes
:?d*
strides
*
data_formatNHWC*
T0*
explicit_paddings
 *
	dilations
*
use_cudnn_on_gpu(*
paddingVALID
?
:gradients_5/decoder_1/layer_0/conv2d_transpose_grad/Conv2DConv2DPgradients_5/decoder_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGraddecoder/layer_0/kernel/read*
data_formatNHWC*
	dilations
*
explicit_paddings
 */
_output_shapes
:?????????d*
T0*
strides
*
paddingVALID*
use_cudnn_on_gpu(
?
(gradients_5/decoder_1/Reshape_grad/ShapeShape@MultivariateNormalDiag/sample/affine_linear_operator/forward/add*
T0*
_output_shapes
:*
out_type0
?
*gradients_5/decoder_1/Reshape_grad/ReshapeReshape:gradients_5/decoder_1/layer_0/conv2d_transpose_grad/Conv2D(gradients_5/decoder_1/Reshape_grad/Shape*'
_output_shapes
:?????????d*
T0*
Tshape0
?
Wgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/ShapeShapeZMultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul*
_output_shapes
:*
out_type0*
T0
?
Ygradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape_1ShapeHencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add*
_output_shapes
:*
T0*
out_type0
?
ggradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/BroadcastGradientArgsBroadcastGradientArgsWgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/ShapeYgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
Ugradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/SumSum*gradients_5/decoder_1/Reshape_grad/Reshapeggradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
?
Ygradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/ReshapeReshapeUgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/SumWgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape*'
_output_shapes
:?????????d*
T0*
Tshape0
?
Wgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Sum_1Sum*gradients_5/decoder_1/Reshape_grad/Reshapeigradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
[gradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1ReshapeWgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Sum_1Ygradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:?????????d
?
qgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/ShapeShapeadd*
T0*
_output_shapes
:*
out_type0
?
sgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape_1Shape%MultivariateNormalDiag/sample/Reshape*
T0*
out_type0*
_output_shapes
:
?
?gradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/BroadcastGradientArgsBroadcastGradientArgsqgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shapesgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
ogradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/MulMulYgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape%MultivariateNormalDiag/sample/Reshape*'
_output_shapes
:?????????d*
T0
?
ogradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/SumSumogradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Mul?gradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
?
sgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/ReshapeReshapeogradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Sumqgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape*
T0*'
_output_shapes
:?????????d*
Tshape0
?
qgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Mul_1MuladdYgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape*'
_output_shapes
:?????????d*
T0
?
qgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Sum_1Sumqgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Mul_1?gradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
?
ugradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape_1Reshapeqgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Sum_1sgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????d
n
gradients_5/add_grad/ShapeShapegen/dense_3/Softplus*
out_type0*
T0*
_output_shapes
:
_
gradients_5/add_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
?
*gradients_5/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/add_grad/Shapegradients_5/add_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_5/add_grad/SumSumsgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape*gradients_5/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
gradients_5/add_grad/ReshapeReshapegradients_5/add_grad/Sumgradients_5/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????d
?
gradients_5/add_grad/Sum_1Sumsgradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape,gradients_5/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
?
gradients_5/add_grad/Reshape_1Reshapegradients_5/add_grad/Sum_1gradients_5/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0

-gradients_5/gen/dense_3/Softplus_grad/SigmoidSigmoidgen/dense_3/BiasAdd*
T0*'
_output_shapes
:?????????d
?
)gradients_5/gen/dense_3/Softplus_grad/mulMulgradients_5/add_grad/Reshape-gradients_5/gen/dense_3/Softplus_grad/Sigmoid*
T0*'
_output_shapes
:?????????d
?
0gradients_5/gen/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_5/gen/dense_3/Softplus_grad/mul*
_output_shapes
:d*
data_formatNHWC*
T0
?
*gradients_5/gen/dense_3/MatMul_grad/MatMulMatMul)gradients_5/gen/dense_3/Softplus_grad/mulgen/dense_3/kernel/read*
T0*
transpose_a( *'
_output_shapes
:????????? *
transpose_b(
?
,gradients_5/gen/dense_3/MatMul_grad/MatMul_1MatMulgen/dense_2/LeakyRelu)gradients_5/gen/dense_3/Softplus_grad/mul*
transpose_b( *
transpose_a(*
_output_shapes

: d*
T0
?
4gradients_5/gen/dense_2/LeakyRelu_grad/LeakyReluGradLeakyReluGrad*gradients_5/gen/dense_3/MatMul_grad/MatMulgen/dense_2/BiasAdd*
alpha%??L>*
T0*'
_output_shapes
:????????? 
?
0gradients_5/gen/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients_5/gen/dense_2/LeakyRelu_grad/LeakyReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
?
*gradients_5/gen/dense_2/MatMul_grad/MatMulMatMul4gradients_5/gen/dense_2/LeakyRelu_grad/LeakyReluGradgen/dense_2/kernel/read*
transpose_b(*
T0*'
_output_shapes
:????????? *
transpose_a( 
?
,gradients_5/gen/dense_2/MatMul_grad/MatMul_1MatMulgen/dense_1/LeakyRelu4gradients_5/gen/dense_2/LeakyRelu_grad/LeakyReluGrad*
transpose_b( *
T0*
_output_shapes

:  *
transpose_a(
?
4gradients_5/gen/dense_1/LeakyRelu_grad/LeakyReluGradLeakyReluGrad*gradients_5/gen/dense_2/MatMul_grad/MatMulgen/dense_1/BiasAdd*'
_output_shapes
:????????? *
alpha%??L>*
T0
?
0gradients_5/gen/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients_5/gen/dense_1/LeakyRelu_grad/LeakyReluGrad*
_output_shapes
: *
T0*
data_formatNHWC
?
*gradients_5/gen/dense_1/MatMul_grad/MatMulMatMul4gradients_5/gen/dense_1/LeakyRelu_grad/LeakyReluGradgen/dense_1/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:????????? 
?
,gradients_5/gen/dense_1/MatMul_grad/MatMul_1MatMulgen/dense/LeakyRelu4gradients_5/gen/dense_1/LeakyRelu_grad/LeakyReluGrad*
_output_shapes

:  *
T0*
transpose_b( *
transpose_a(
?
2gradients_5/gen/dense/LeakyRelu_grad/LeakyReluGradLeakyReluGrad*gradients_5/gen/dense_1/MatMul_grad/MatMulgen/dense/BiasAdd*
T0*'
_output_shapes
:????????? *
alpha%??L>
?
.gradients_5/gen/dense/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients_5/gen/dense/LeakyRelu_grad/LeakyReluGrad*
data_formatNHWC*
T0*
_output_shapes
: 
?
(gradients_5/gen/dense/MatMul_grad/MatMulMatMul2gradients_5/gen/dense/LeakyRelu_grad/LeakyReluGradgen/dense/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:?????????f
?
*gradients_5/gen/dense/MatMul_grad/MatMul_1MatMul
gen/concat2gradients_5/gen/dense/LeakyRelu_grad/LeakyReluGrad*
_output_shapes

:f *
T0*
transpose_b( *
transpose_a(
b
 gradients_5/gen/concat_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :

gradients_5/gen/concat_grad/modFloorModgen/concat/axis gradients_5/gen/concat_grad/Rank*
_output_shapes
: *
T0
r
!gradients_5/gen/concat_grad/ShapeShapegen/random_normal*
T0*
out_type0*
_output_shapes
:
?
"gradients_5/gen/concat_grad/ShapeNShapeNgen/random_normalHencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add* 
_output_shapes
::*
N*
out_type0*
T0
?
(gradients_5/gen/concat_grad/ConcatOffsetConcatOffsetgradients_5/gen/concat_grad/mod"gradients_5/gen/concat_grad/ShapeN$gradients_5/gen/concat_grad/ShapeN:1* 
_output_shapes
::*
N
?
!gradients_5/gen/concat_grad/SliceSlice(gradients_5/gen/dense/MatMul_grad/MatMul(gradients_5/gen/concat_grad/ConcatOffset"gradients_5/gen/concat_grad/ShapeN*
T0*
Index0*'
_output_shapes
:?????????
?
#gradients_5/gen/concat_grad/Slice_1Slice(gradients_5/gen/dense/MatMul_grad/MatMul*gradients_5/gen/concat_grad/ConcatOffset:1$gradients_5/gen/concat_grad/ShapeN:1*
Index0*'
_output_shapes
:?????????d*
T0
?
gradients_5/AddN_3AddN[gradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1#gradients_5/gen/concat_grad/Slice_1*n
_classd
b`loc:@gradients_5/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1*'
_output_shapes
:?????????d*
N*
T0
?
_gradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/ShapeShapebencoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul*
_output_shapes
:*
out_type0*
T0
?
agradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape_1Shapeencoder/dense/BiasAdd*
T0*
_output_shapes
:*
out_type0
?
ogradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shapeagradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
]gradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/SumSumgradients_5/AddN_3ogradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
?
agradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/ReshapeReshape]gradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Sum_gradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape*'
_output_shapes
:?????????d*
Tshape0*
T0
?
_gradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Sum_1Sumgradients_5/AddN_3qgradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
?
cgradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1Reshape_gradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Sum_1agradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Shape_1*'
_output_shapes
:?????????d*
T0*
Tshape0
?
ygradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/ShapeShapeencoder/dense_1/Softplus*
T0*
_output_shapes
:*
out_type0
?
{gradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape_1Shape-encoder/MultivariateNormalDiag/sample/Reshape*
out_type0*
T0*
_output_shapes
:
?
?gradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/BroadcastGradientArgsBroadcastGradientArgsygradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape{gradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
wgradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/MulMulagradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape-encoder/MultivariateNormalDiag/sample/Reshape*
T0*'
_output_shapes
:?????????d
?
wgradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/SumSumwgradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Mul?gradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
{gradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/ReshapeReshapewgradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Sumygradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape*
Tshape0*
T0*'
_output_shapes
:?????????d
?
ygradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Mul_1Mulencoder/dense_1/Softplusagradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape*
T0*'
_output_shapes
:?????????d
?
ygradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Sum_1Sumygradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Mul_1?gradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
}gradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape_1Reshapeygradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Sum_1{gradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Shape_1*
Tshape0*'
_output_shapes
:?????????d*
T0
?
2gradients_5/encoder/dense/BiasAdd_grad/BiasAddGradBiasAddGradcgradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1*
data_formatNHWC*
_output_shapes
:d*
T0
?
1gradients_5/encoder/dense_1/Softplus_grad/SigmoidSigmoidencoder/dense_1/BiasAdd*
T0*'
_output_shapes
:?????????d
?
-gradients_5/encoder/dense_1/Softplus_grad/mulMul{gradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/LinearOperatorDiag/matvec/mul_grad/Reshape1gradients_5/encoder/dense_1/Softplus_grad/Sigmoid*'
_output_shapes
:?????????d*
T0
?
,gradients_5/encoder/dense/MatMul_grad/MatMulMatMulcgradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1encoder/dense/kernel/read*
transpose_b(*(
_output_shapes
:??????????*
transpose_a( *
T0
?
.gradients_5/encoder/dense/MatMul_grad/MatMul_1MatMulencoder/flatten/Reshapecgradients_5/encoder/MultivariateNormalDiag/sample/affine_linear_operator/forward/add_grad/Reshape_1*
transpose_a(*
transpose_b( *
T0*
_output_shapes
:	?d
?
4gradients_5/encoder/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients_5/encoder/dense_1/Softplus_grad/mul*
data_formatNHWC*
_output_shapes
:d*
T0
?
.gradients_5/encoder/dense_1/MatMul_grad/MatMulMatMul-gradients_5/encoder/dense_1/Softplus_grad/mulencoder/dense_1/kernel/read*
transpose_a( *
transpose_b(*(
_output_shapes
:??????????*
T0
?
0gradients_5/encoder/dense_1/MatMul_grad/MatMul_1MatMulencoder/flatten/Reshape-gradients_5/encoder/dense_1/Softplus_grad/mul*
_output_shapes
:	?d*
transpose_a(*
T0*
transpose_b( 
?
gradients_5/AddN_4AddN,gradients_5/encoder/dense/MatMul_grad/MatMul.gradients_5/encoder/dense_1/MatMul_grad/MatMul*
N*(
_output_shapes
:??????????*
T0*?
_class5
31loc:@gradients_5/encoder/dense/MatMul_grad/MatMul
?
.gradients_5/encoder/flatten/Reshape_grad/ShapeShapeencoder/max_pooling2d_1/MaxPool*
T0*
_output_shapes
:*
out_type0
?
0gradients_5/encoder/flatten/Reshape_grad/ReshapeReshapegradients_5/AddN_4.gradients_5/encoder/flatten/Reshape_grad/Shape*/
_output_shapes
:?????????2*
Tshape0*
T0
?
<gradients_5/encoder/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGradencoder/conv2d_1/Reluencoder/max_pooling2d_1/MaxPool0gradients_5/encoder/flatten/Reshape_grad/Reshape*
strides
*/
_output_shapes
:?????????2*
paddingVALID*
ksize
*
T0*
data_formatNHWC
?
/gradients_5/encoder/conv2d_1/Relu_grad/ReluGradReluGrad<gradients_5/encoder/max_pooling2d_1/MaxPool_grad/MaxPoolGradencoder/conv2d_1/Relu*/
_output_shapes
:?????????2*
T0
?
5gradients_5/encoder/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad/gradients_5/encoder/conv2d_1/Relu_grad/ReluGrad*
_output_shapes
:2*
T0*
data_formatNHWC
?
/gradients_5/encoder/conv2d_1/Conv2D_grad/ShapeNShapeNencoder/max_pooling2d/MaxPoolencoder/conv2d_1/kernel/read*
T0* 
_output_shapes
::*
out_type0*
N
?
<gradients_5/encoder/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput/gradients_5/encoder/conv2d_1/Conv2D_grad/ShapeNencoder/conv2d_1/kernel/read/gradients_5/encoder/conv2d_1/Relu_grad/ReluGrad*
explicit_paddings
 *
strides
*
data_formatNHWC*/
_output_shapes
:?????????*
	dilations
*
use_cudnn_on_gpu(*
paddingVALID*
T0
?
=gradients_5/encoder/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterencoder/max_pooling2d/MaxPool1gradients_5/encoder/conv2d_1/Conv2D_grad/ShapeN:1/gradients_5/encoder/conv2d_1/Relu_grad/ReluGrad*
strides
*
use_cudnn_on_gpu(*
data_formatNHWC*
	dilations
*
explicit_paddings
 *&
_output_shapes
:2*
T0*
paddingVALID
?
:gradients_5/encoder/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradencoder/conv2d/Reluencoder/max_pooling2d/MaxPool<gradients_5/encoder/conv2d_1/Conv2D_grad/Conv2DBackpropInput*
T0*
data_formatNHWC*
ksize
*
strides
*/
_output_shapes
:?????????*
paddingVALID
?
-gradients_5/encoder/conv2d/Relu_grad/ReluGradReluGrad:gradients_5/encoder/max_pooling2d/MaxPool_grad/MaxPoolGradencoder/conv2d/Relu*
T0*/
_output_shapes
:?????????
?
3gradients_5/encoder/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients_5/encoder/conv2d/Relu_grad/ReluGrad*
T0*
_output_shapes
:*
data_formatNHWC
?
-gradients_5/encoder/conv2d/Conv2D_grad/ShapeNShapeNencoder/Reshapeencoder/conv2d/kernel/read*
N* 
_output_shapes
::*
out_type0*
T0
?
:gradients_5/encoder/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput-gradients_5/encoder/conv2d/Conv2D_grad/ShapeNencoder/conv2d/kernel/read-gradients_5/encoder/conv2d/Relu_grad/ReluGrad*
data_formatNHWC*/
_output_shapes
:?????????*
paddingVALID*
use_cudnn_on_gpu(*
T0*
	dilations
*
explicit_paddings
 *
strides

?
;gradients_5/encoder/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterencoder/Reshape/gradients_5/encoder/conv2d/Conv2D_grad/ShapeN:1-gradients_5/encoder/conv2d/Relu_grad/ReluGrad*
use_cudnn_on_gpu(*
strides
*&
_output_shapes
:*
paddingVALID*
data_formatNHWC*
explicit_paddings
 *
T0*
	dilations

g
&gradients_5/encoder/Reshape_grad/ShapeShapeX*
T0*
_output_shapes
:*
out_type0
?
(gradients_5/encoder/Reshape_grad/ReshapeReshape:gradients_5/encoder/conv2d/Conv2D_grad/Conv2DBackpropInput&gradients_5/encoder/Reshape_grad/Shape*(
_output_shapes
:??????????*
Tshape0*
T0
?
gradients_5/AddN_5AddN%gradients_5/disc/Reshape_grad/Reshape(gradients_5/encoder/Reshape_grad/Reshape*
T0*
N*(
_output_shapes
:??????????*8
_class.
,*loc:@gradients_5/disc/Reshape_grad/Reshape
S
SignSigngradients_5/AddN_5*
T0*(
_output_shapes
:??????????
B
mul_6MulPlaceholderSign*
_output_shapes
:*
T0
:
add_27AddXmul_6*
_output_shapes
:*
T0
,
group_deps_2NoOp^Adam_1^group_deps_1
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
?
ArgMaxArgMaxdisc/dense_1/BiasAddArgMax/dimension*
T0*

Tidx0*#
_output_shapes
:?????????*
output_type0	
T
ArgMax_1/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
v
ArgMax_1ArgMaxYArgMax_1/dimension*
output_type0	*

Tidx0*#
_output_shapes
:?????????*
T0
P
Equal_1EqualArgMaxArgMax_1*#
_output_shapes
:?????????*
T0	
b
CastCastEqual_1*#
_output_shapes
:?????????*
Truncate( *

DstT0*

SrcT0

`
Reshape_1/shapeConst*
dtype0*
valueB"????   *
_output_shapes
:
k
	Reshape_1ReshapeCastReshape_1/shape*
Tshape0*
T0*'
_output_shapes
:?????????
Y
Const_15Const*
_output_shapes
:*
dtype0*
valueB"       
b
Mean_13Mean	Reshape_1Const_15*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
Z
Sum_13/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
{
Sum_13SumExpSum_13/reduction_indices*

Tidx0*'
_output_shapes
:?????????*
	keep_dims(*
T0
Y
Const_16Const*
_output_shapes
:*
valueB"       *
dtype0
_
Mean_14MeanSum_13Const_16*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Z
Sum_14/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0
{
Sum_14SumExpSum_14/reduction_indices*
T0*
	keep_dims(*

Tidx0*'
_output_shapes
:?????????
Q
mul_7MulSum_14	Reshape_1*
T0*'
_output_shapes
:?????????
Y
Const_17Const*
valueB"       *
dtype0*
_output_shapes
:
\
Sum_15Summul_7Const_17*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
M
add_28/yConst*
valueB
 *?<*
_output_shapes
: *
dtype0
T
add_28Add	Reshape_1add_28/y*
T0*'
_output_shapes
:?????????
Y
Const_18Const*
_output_shapes
:*
valueB"       *
dtype0
]
Sum_16Sumadd_28Const_18*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
E
	truediv_2RealDivSum_15Sum_16*
_output_shapes
: *
T0
Z
Sum_17/reduction_indicesConst*
value	B :*
_output_shapes
: *
dtype0
{
Sum_17SumExpSum_17/reduction_indices*
T0*

Tidx0*'
_output_shapes
:?????????*
	keep_dims(
M
sub_12/xConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
T
sub_12Subsub_12/x	Reshape_1*
T0*'
_output_shapes
:?????????
N
mul_8MulSum_17sub_12*'
_output_shapes
:?????????*
T0
Y
Const_19Const*
dtype0*
_output_shapes
:*
valueB"       
\
Sum_18Summul_8Const_19*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
M
sub_13/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
T
sub_13Subsub_13/x	Reshape_1*
T0*'
_output_shapes
:?????????
D
AbsAbssub_13*'
_output_shapes
:?????????*
T0
Y
Const_20Const*
dtype0*
valueB"       *
_output_shapes
:
Z
Sum_19SumAbsConst_20*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
M
add_29/yConst*
valueB
 *?<*
_output_shapes
: *
dtype0
@
add_29AddSum_19add_29/y*
_output_shapes
: *
T0
E
	truediv_3RealDivSum_18add_29*
_output_shapes
: *
T0
?*
initNoOp^beta1_power/Assign^beta1_power_1/Assign^beta2_power/Assign^beta2_power_1/Assign-^decoder/batch_normalization/beta/Adam/Assign/^decoder/batch_normalization/beta/Adam_1/Assign(^decoder/batch_normalization/beta/Assign.^decoder/batch_normalization/gamma/Adam/Assign0^decoder/batch_normalization/gamma/Adam_1/Assign)^decoder/batch_normalization/gamma/Assign/^decoder/batch_normalization/moving_mean/Assign3^decoder/batch_normalization/moving_variance/Assign/^decoder/batch_normalization_1/beta/Adam/Assign1^decoder/batch_normalization_1/beta/Adam_1/Assign*^decoder/batch_normalization_1/beta/Assign0^decoder/batch_normalization_1/gamma/Adam/Assign2^decoder/batch_normalization_1/gamma/Adam_1/Assign+^decoder/batch_normalization_1/gamma/Assign1^decoder/batch_normalization_1/moving_mean/Assign5^decoder/batch_normalization_1/moving_variance/Assign!^decoder/layer_0/bias/Adam/Assign#^decoder/layer_0/bias/Adam_1/Assign^decoder/layer_0/bias/Assign#^decoder/layer_0/kernel/Adam/Assign%^decoder/layer_0/kernel/Adam_1/Assign^decoder/layer_0/kernel/Assign!^decoder/layer_1/bias/Adam/Assign#^decoder/layer_1/bias/Adam_1/Assign^decoder/layer_1/bias/Assign#^decoder/layer_1/kernel/Adam/Assign%^decoder/layer_1/kernel/Adam_1/Assign^decoder/layer_1/kernel/Assign!^decoder/layer_2/bias/Adam/Assign#^decoder/layer_2/bias/Adam_1/Assign^decoder/layer_2/bias/Assign#^decoder/layer_2/kernel/Adam/Assign%^decoder/layer_2/kernel/Adam_1/Assign^decoder/layer_2/kernel/Assign^disc/conv2d/bias/Adam/Assign^disc/conv2d/bias/Adam_1/Assign^disc/conv2d/bias/Assign^disc/conv2d/kernel/Adam/Assign!^disc/conv2d/kernel/Adam_1/Assign^disc/conv2d/kernel/Assign^disc/conv2d_1/bias/Adam/Assign!^disc/conv2d_1/bias/Adam_1/Assign^disc/conv2d_1/bias/Assign!^disc/conv2d_1/kernel/Adam/Assign#^disc/conv2d_1/kernel/Adam_1/Assign^disc/conv2d_1/kernel/Assign^disc/dense/bias/Adam/Assign^disc/dense/bias/Adam_1/Assign^disc/dense/bias/Assign^disc/dense/kernel/Adam/Assign ^disc/dense/kernel/Adam_1/Assign^disc/dense/kernel/Assign^disc/dense_1/bias/Adam/Assign ^disc/dense_1/bias/Adam_1/Assign^disc/dense_1/bias/Assign ^disc/dense_1/kernel/Adam/Assign"^disc/dense_1/kernel/Adam_1/Assign^disc/dense_1/kernel/Assign^disc0/conv2d/bias/Assign!^disc0/conv2d/bias/RMSProp/Assign#^disc0/conv2d/bias/RMSProp_1/Assign^disc0/conv2d/kernel/Assign#^disc0/conv2d/kernel/RMSProp/Assign%^disc0/conv2d/kernel/RMSProp_1/Assign^disc0/conv2d_1/bias/Assign#^disc0/conv2d_1/bias/RMSProp/Assign%^disc0/conv2d_1/bias/RMSProp_1/Assign^disc0/conv2d_1/kernel/Assign%^disc0/conv2d_1/kernel/RMSProp/Assign'^disc0/conv2d_1/kernel/RMSProp_1/Assign^disc0/dense/bias/Assign ^disc0/dense/bias/RMSProp/Assign"^disc0/dense/bias/RMSProp_1/Assign^disc0/dense/kernel/Assign"^disc0/dense/kernel/RMSProp/Assign$^disc0/dense/kernel/RMSProp_1/Assign^disc0/dense_1/bias/Assign"^disc0/dense_1/bias/RMSProp/Assign$^disc0/dense_1/bias/RMSProp_1/Assign^disc0/dense_1/kernel/Assign$^disc0/dense_1/kernel/RMSProp/Assign&^disc0/dense_1/kernel/RMSProp_1/Assign^diz/dense/bias/Assign^diz/dense/bias/RMSProp/Assign ^diz/dense/bias/RMSProp_1/Assign^diz/dense/kernel/Assign ^diz/dense/kernel/RMSProp/Assign"^diz/dense/kernel/RMSProp_1/Assign^diz/dense_1/bias/Assign ^diz/dense_1/bias/RMSProp/Assign"^diz/dense_1/bias/RMSProp_1/Assign^diz/dense_1/kernel/Assign"^diz/dense_1/kernel/RMSProp/Assign$^diz/dense_1/kernel/RMSProp_1/Assign^diz/dense_2/bias/Assign ^diz/dense_2/bias/RMSProp/Assign"^diz/dense_2/bias/RMSProp_1/Assign^diz/dense_2/kernel/Assign"^diz/dense_2/kernel/RMSProp/Assign$^diz/dense_2/kernel/RMSProp_1/Assign^diz/dense_3/bias/Assign ^diz/dense_3/bias/RMSProp/Assign"^diz/dense_3/bias/RMSProp_1/Assign^diz/dense_3/kernel/Assign"^diz/dense_3/kernel/RMSProp/Assign$^diz/dense_3/kernel/RMSProp_1/Assign ^encoder/conv2d/bias/Adam/Assign"^encoder/conv2d/bias/Adam_1/Assign^encoder/conv2d/bias/Assign"^encoder/conv2d/kernel/Adam/Assign$^encoder/conv2d/kernel/Adam_1/Assign^encoder/conv2d/kernel/Assign"^encoder/conv2d_1/bias/Adam/Assign$^encoder/conv2d_1/bias/Adam_1/Assign^encoder/conv2d_1/bias/Assign$^encoder/conv2d_1/kernel/Adam/Assign&^encoder/conv2d_1/kernel/Adam_1/Assign^encoder/conv2d_1/kernel/Assign^encoder/dense/bias/Adam/Assign!^encoder/dense/bias/Adam_1/Assign^encoder/dense/bias/Assign!^encoder/dense/kernel/Adam/Assign#^encoder/dense/kernel/Adam_1/Assign^encoder/dense/kernel/Assign!^encoder/dense_1/bias/Adam/Assign#^encoder/dense_1/bias/Adam_1/Assign^encoder/dense_1/bias/Assign#^encoder/dense_1/kernel/Adam/Assign%^encoder/dense_1/kernel/Adam_1/Assign^encoder/dense_1/kernel/Assign^gen/dense/bias/Assign^gen/dense/bias/RMSProp/Assign ^gen/dense/bias/RMSProp_1/Assign^gen/dense/kernel/Assign ^gen/dense/kernel/RMSProp/Assign"^gen/dense/kernel/RMSProp_1/Assign^gen/dense_1/bias/Assign ^gen/dense_1/bias/RMSProp/Assign"^gen/dense_1/bias/RMSProp_1/Assign^gen/dense_1/kernel/Assign"^gen/dense_1/kernel/RMSProp/Assign$^gen/dense_1/kernel/RMSProp_1/Assign^gen/dense_2/bias/Assign ^gen/dense_2/bias/RMSProp/Assign"^gen/dense_2/bias/RMSProp_1/Assign^gen/dense_2/kernel/Assign"^gen/dense_2/kernel/RMSProp/Assign$^gen/dense_2/kernel/RMSProp_1/Assign^gen/dense_3/bias/Assign ^gen/dense_3/bias/RMSProp/Assign"^gen/dense_3/bias/RMSProp_1/Assign^gen/dense_3/kernel/Assign"^gen/dense_3/kernel/RMSProp/Assign$^gen/dense_3/kernel/RMSProp_1/Assign
Y
save/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
_output_shapes
: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
shape: *
dtype0
?
save/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_821a7760356b4e0a801be7e51e4abc00/part*
dtype0
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
?!
save/SaveV2/tensor_namesConst*
dtype0*?!
value? B? ?Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1B decoder/batch_normalization/betaB%decoder/batch_normalization/beta/AdamB'decoder/batch_normalization/beta/Adam_1B!decoder/batch_normalization/gammaB&decoder/batch_normalization/gamma/AdamB(decoder/batch_normalization/gamma/Adam_1B'decoder/batch_normalization/moving_meanB+decoder/batch_normalization/moving_varianceB"decoder/batch_normalization_1/betaB'decoder/batch_normalization_1/beta/AdamB)decoder/batch_normalization_1/beta/Adam_1B#decoder/batch_normalization_1/gammaB(decoder/batch_normalization_1/gamma/AdamB*decoder/batch_normalization_1/gamma/Adam_1B)decoder/batch_normalization_1/moving_meanB-decoder/batch_normalization_1/moving_varianceBdecoder/layer_0/biasBdecoder/layer_0/bias/AdamBdecoder/layer_0/bias/Adam_1Bdecoder/layer_0/kernelBdecoder/layer_0/kernel/AdamBdecoder/layer_0/kernel/Adam_1Bdecoder/layer_1/biasBdecoder/layer_1/bias/AdamBdecoder/layer_1/bias/Adam_1Bdecoder/layer_1/kernelBdecoder/layer_1/kernel/AdamBdecoder/layer_1/kernel/Adam_1Bdecoder/layer_2/biasBdecoder/layer_2/bias/AdamBdecoder/layer_2/bias/Adam_1Bdecoder/layer_2/kernelBdecoder/layer_2/kernel/AdamBdecoder/layer_2/kernel/Adam_1Bdisc/conv2d/biasBdisc/conv2d/bias/AdamBdisc/conv2d/bias/Adam_1Bdisc/conv2d/kernelBdisc/conv2d/kernel/AdamBdisc/conv2d/kernel/Adam_1Bdisc/conv2d_1/biasBdisc/conv2d_1/bias/AdamBdisc/conv2d_1/bias/Adam_1Bdisc/conv2d_1/kernelBdisc/conv2d_1/kernel/AdamBdisc/conv2d_1/kernel/Adam_1Bdisc/dense/biasBdisc/dense/bias/AdamBdisc/dense/bias/Adam_1Bdisc/dense/kernelBdisc/dense/kernel/AdamBdisc/dense/kernel/Adam_1Bdisc/dense_1/biasBdisc/dense_1/bias/AdamBdisc/dense_1/bias/Adam_1Bdisc/dense_1/kernelBdisc/dense_1/kernel/AdamBdisc/dense_1/kernel/Adam_1Bdisc0/conv2d/biasBdisc0/conv2d/bias/RMSPropBdisc0/conv2d/bias/RMSProp_1Bdisc0/conv2d/kernelBdisc0/conv2d/kernel/RMSPropBdisc0/conv2d/kernel/RMSProp_1Bdisc0/conv2d_1/biasBdisc0/conv2d_1/bias/RMSPropBdisc0/conv2d_1/bias/RMSProp_1Bdisc0/conv2d_1/kernelBdisc0/conv2d_1/kernel/RMSPropBdisc0/conv2d_1/kernel/RMSProp_1Bdisc0/dense/biasBdisc0/dense/bias/RMSPropBdisc0/dense/bias/RMSProp_1Bdisc0/dense/kernelBdisc0/dense/kernel/RMSPropBdisc0/dense/kernel/RMSProp_1Bdisc0/dense_1/biasBdisc0/dense_1/bias/RMSPropBdisc0/dense_1/bias/RMSProp_1Bdisc0/dense_1/kernelBdisc0/dense_1/kernel/RMSPropBdisc0/dense_1/kernel/RMSProp_1Bdiz/dense/biasBdiz/dense/bias/RMSPropBdiz/dense/bias/RMSProp_1Bdiz/dense/kernelBdiz/dense/kernel/RMSPropBdiz/dense/kernel/RMSProp_1Bdiz/dense_1/biasBdiz/dense_1/bias/RMSPropBdiz/dense_1/bias/RMSProp_1Bdiz/dense_1/kernelBdiz/dense_1/kernel/RMSPropBdiz/dense_1/kernel/RMSProp_1Bdiz/dense_2/biasBdiz/dense_2/bias/RMSPropBdiz/dense_2/bias/RMSProp_1Bdiz/dense_2/kernelBdiz/dense_2/kernel/RMSPropBdiz/dense_2/kernel/RMSProp_1Bdiz/dense_3/biasBdiz/dense_3/bias/RMSPropBdiz/dense_3/bias/RMSProp_1Bdiz/dense_3/kernelBdiz/dense_3/kernel/RMSPropBdiz/dense_3/kernel/RMSProp_1Bencoder/conv2d/biasBencoder/conv2d/bias/AdamBencoder/conv2d/bias/Adam_1Bencoder/conv2d/kernelBencoder/conv2d/kernel/AdamBencoder/conv2d/kernel/Adam_1Bencoder/conv2d_1/biasBencoder/conv2d_1/bias/AdamBencoder/conv2d_1/bias/Adam_1Bencoder/conv2d_1/kernelBencoder/conv2d_1/kernel/AdamBencoder/conv2d_1/kernel/Adam_1Bencoder/dense/biasBencoder/dense/bias/AdamBencoder/dense/bias/Adam_1Bencoder/dense/kernelBencoder/dense/kernel/AdamBencoder/dense/kernel/Adam_1Bencoder/dense_1/biasBencoder/dense_1/bias/AdamBencoder/dense_1/bias/Adam_1Bencoder/dense_1/kernelBencoder/dense_1/kernel/AdamBencoder/dense_1/kernel/Adam_1Bgen/dense/biasBgen/dense/bias/RMSPropBgen/dense/bias/RMSProp_1Bgen/dense/kernelBgen/dense/kernel/RMSPropBgen/dense/kernel/RMSProp_1Bgen/dense_1/biasBgen/dense_1/bias/RMSPropBgen/dense_1/bias/RMSProp_1Bgen/dense_1/kernelBgen/dense_1/kernel/RMSPropBgen/dense_1/kernel/RMSProp_1Bgen/dense_2/biasBgen/dense_2/bias/RMSPropBgen/dense_2/bias/RMSProp_1Bgen/dense_2/kernelBgen/dense_2/kernel/RMSPropBgen/dense_2/kernel/RMSProp_1Bgen/dense_3/biasBgen/dense_3/bias/RMSPropBgen/dense_3/bias/RMSProp_1Bgen/dense_3/kernelBgen/dense_3/kernel/RMSPropBgen/dense_3/kernel/RMSProp_1*
_output_shapes	
:?
?
save/SaveV2/shape_and_slicesConst*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes	
:?
?#
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1 decoder/batch_normalization/beta%decoder/batch_normalization/beta/Adam'decoder/batch_normalization/beta/Adam_1!decoder/batch_normalization/gamma&decoder/batch_normalization/gamma/Adam(decoder/batch_normalization/gamma/Adam_1'decoder/batch_normalization/moving_mean+decoder/batch_normalization/moving_variance"decoder/batch_normalization_1/beta'decoder/batch_normalization_1/beta/Adam)decoder/batch_normalization_1/beta/Adam_1#decoder/batch_normalization_1/gamma(decoder/batch_normalization_1/gamma/Adam*decoder/batch_normalization_1/gamma/Adam_1)decoder/batch_normalization_1/moving_mean-decoder/batch_normalization_1/moving_variancedecoder/layer_0/biasdecoder/layer_0/bias/Adamdecoder/layer_0/bias/Adam_1decoder/layer_0/kerneldecoder/layer_0/kernel/Adamdecoder/layer_0/kernel/Adam_1decoder/layer_1/biasdecoder/layer_1/bias/Adamdecoder/layer_1/bias/Adam_1decoder/layer_1/kerneldecoder/layer_1/kernel/Adamdecoder/layer_1/kernel/Adam_1decoder/layer_2/biasdecoder/layer_2/bias/Adamdecoder/layer_2/bias/Adam_1decoder/layer_2/kerneldecoder/layer_2/kernel/Adamdecoder/layer_2/kernel/Adam_1disc/conv2d/biasdisc/conv2d/bias/Adamdisc/conv2d/bias/Adam_1disc/conv2d/kerneldisc/conv2d/kernel/Adamdisc/conv2d/kernel/Adam_1disc/conv2d_1/biasdisc/conv2d_1/bias/Adamdisc/conv2d_1/bias/Adam_1disc/conv2d_1/kerneldisc/conv2d_1/kernel/Adamdisc/conv2d_1/kernel/Adam_1disc/dense/biasdisc/dense/bias/Adamdisc/dense/bias/Adam_1disc/dense/kerneldisc/dense/kernel/Adamdisc/dense/kernel/Adam_1disc/dense_1/biasdisc/dense_1/bias/Adamdisc/dense_1/bias/Adam_1disc/dense_1/kerneldisc/dense_1/kernel/Adamdisc/dense_1/kernel/Adam_1disc0/conv2d/biasdisc0/conv2d/bias/RMSPropdisc0/conv2d/bias/RMSProp_1disc0/conv2d/kerneldisc0/conv2d/kernel/RMSPropdisc0/conv2d/kernel/RMSProp_1disc0/conv2d_1/biasdisc0/conv2d_1/bias/RMSPropdisc0/conv2d_1/bias/RMSProp_1disc0/conv2d_1/kerneldisc0/conv2d_1/kernel/RMSPropdisc0/conv2d_1/kernel/RMSProp_1disc0/dense/biasdisc0/dense/bias/RMSPropdisc0/dense/bias/RMSProp_1disc0/dense/kerneldisc0/dense/kernel/RMSPropdisc0/dense/kernel/RMSProp_1disc0/dense_1/biasdisc0/dense_1/bias/RMSPropdisc0/dense_1/bias/RMSProp_1disc0/dense_1/kerneldisc0/dense_1/kernel/RMSPropdisc0/dense_1/kernel/RMSProp_1diz/dense/biasdiz/dense/bias/RMSPropdiz/dense/bias/RMSProp_1diz/dense/kerneldiz/dense/kernel/RMSPropdiz/dense/kernel/RMSProp_1diz/dense_1/biasdiz/dense_1/bias/RMSPropdiz/dense_1/bias/RMSProp_1diz/dense_1/kerneldiz/dense_1/kernel/RMSPropdiz/dense_1/kernel/RMSProp_1diz/dense_2/biasdiz/dense_2/bias/RMSPropdiz/dense_2/bias/RMSProp_1diz/dense_2/kerneldiz/dense_2/kernel/RMSPropdiz/dense_2/kernel/RMSProp_1diz/dense_3/biasdiz/dense_3/bias/RMSPropdiz/dense_3/bias/RMSProp_1diz/dense_3/kerneldiz/dense_3/kernel/RMSPropdiz/dense_3/kernel/RMSProp_1encoder/conv2d/biasencoder/conv2d/bias/Adamencoder/conv2d/bias/Adam_1encoder/conv2d/kernelencoder/conv2d/kernel/Adamencoder/conv2d/kernel/Adam_1encoder/conv2d_1/biasencoder/conv2d_1/bias/Adamencoder/conv2d_1/bias/Adam_1encoder/conv2d_1/kernelencoder/conv2d_1/kernel/Adamencoder/conv2d_1/kernel/Adam_1encoder/dense/biasencoder/dense/bias/Adamencoder/dense/bias/Adam_1encoder/dense/kernelencoder/dense/kernel/Adamencoder/dense/kernel/Adam_1encoder/dense_1/biasencoder/dense_1/bias/Adamencoder/dense_1/bias/Adam_1encoder/dense_1/kernelencoder/dense_1/kernel/Adamencoder/dense_1/kernel/Adam_1gen/dense/biasgen/dense/bias/RMSPropgen/dense/bias/RMSProp_1gen/dense/kernelgen/dense/kernel/RMSPropgen/dense/kernel/RMSProp_1gen/dense_1/biasgen/dense_1/bias/RMSPropgen/dense_1/bias/RMSProp_1gen/dense_1/kernelgen/dense_1/kernel/RMSPropgen/dense_1/kernel/RMSProp_1gen/dense_2/biasgen/dense_2/bias/RMSPropgen/dense_2/bias/RMSProp_1gen/dense_2/kernelgen/dense_2/kernel/RMSPropgen/dense_2/kernel/RMSProp_1gen/dense_3/biasgen/dense_3/bias/RMSPropgen/dense_3/bias/RMSProp_1gen/dense_3/kernelgen/dense_3/kernel/RMSPropgen/dense_3/kernel/RMSProp_1*?
dtypes?
?2?
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*

axis *
T0*
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
_output_shapes
: *
T0
?!
save/RestoreV2/tensor_namesConst*
_output_shapes	
:?*?!
value? B? ?Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1B decoder/batch_normalization/betaB%decoder/batch_normalization/beta/AdamB'decoder/batch_normalization/beta/Adam_1B!decoder/batch_normalization/gammaB&decoder/batch_normalization/gamma/AdamB(decoder/batch_normalization/gamma/Adam_1B'decoder/batch_normalization/moving_meanB+decoder/batch_normalization/moving_varianceB"decoder/batch_normalization_1/betaB'decoder/batch_normalization_1/beta/AdamB)decoder/batch_normalization_1/beta/Adam_1B#decoder/batch_normalization_1/gammaB(decoder/batch_normalization_1/gamma/AdamB*decoder/batch_normalization_1/gamma/Adam_1B)decoder/batch_normalization_1/moving_meanB-decoder/batch_normalization_1/moving_varianceBdecoder/layer_0/biasBdecoder/layer_0/bias/AdamBdecoder/layer_0/bias/Adam_1Bdecoder/layer_0/kernelBdecoder/layer_0/kernel/AdamBdecoder/layer_0/kernel/Adam_1Bdecoder/layer_1/biasBdecoder/layer_1/bias/AdamBdecoder/layer_1/bias/Adam_1Bdecoder/layer_1/kernelBdecoder/layer_1/kernel/AdamBdecoder/layer_1/kernel/Adam_1Bdecoder/layer_2/biasBdecoder/layer_2/bias/AdamBdecoder/layer_2/bias/Adam_1Bdecoder/layer_2/kernelBdecoder/layer_2/kernel/AdamBdecoder/layer_2/kernel/Adam_1Bdisc/conv2d/biasBdisc/conv2d/bias/AdamBdisc/conv2d/bias/Adam_1Bdisc/conv2d/kernelBdisc/conv2d/kernel/AdamBdisc/conv2d/kernel/Adam_1Bdisc/conv2d_1/biasBdisc/conv2d_1/bias/AdamBdisc/conv2d_1/bias/Adam_1Bdisc/conv2d_1/kernelBdisc/conv2d_1/kernel/AdamBdisc/conv2d_1/kernel/Adam_1Bdisc/dense/biasBdisc/dense/bias/AdamBdisc/dense/bias/Adam_1Bdisc/dense/kernelBdisc/dense/kernel/AdamBdisc/dense/kernel/Adam_1Bdisc/dense_1/biasBdisc/dense_1/bias/AdamBdisc/dense_1/bias/Adam_1Bdisc/dense_1/kernelBdisc/dense_1/kernel/AdamBdisc/dense_1/kernel/Adam_1Bdisc0/conv2d/biasBdisc0/conv2d/bias/RMSPropBdisc0/conv2d/bias/RMSProp_1Bdisc0/conv2d/kernelBdisc0/conv2d/kernel/RMSPropBdisc0/conv2d/kernel/RMSProp_1Bdisc0/conv2d_1/biasBdisc0/conv2d_1/bias/RMSPropBdisc0/conv2d_1/bias/RMSProp_1Bdisc0/conv2d_1/kernelBdisc0/conv2d_1/kernel/RMSPropBdisc0/conv2d_1/kernel/RMSProp_1Bdisc0/dense/biasBdisc0/dense/bias/RMSPropBdisc0/dense/bias/RMSProp_1Bdisc0/dense/kernelBdisc0/dense/kernel/RMSPropBdisc0/dense/kernel/RMSProp_1Bdisc0/dense_1/biasBdisc0/dense_1/bias/RMSPropBdisc0/dense_1/bias/RMSProp_1Bdisc0/dense_1/kernelBdisc0/dense_1/kernel/RMSPropBdisc0/dense_1/kernel/RMSProp_1Bdiz/dense/biasBdiz/dense/bias/RMSPropBdiz/dense/bias/RMSProp_1Bdiz/dense/kernelBdiz/dense/kernel/RMSPropBdiz/dense/kernel/RMSProp_1Bdiz/dense_1/biasBdiz/dense_1/bias/RMSPropBdiz/dense_1/bias/RMSProp_1Bdiz/dense_1/kernelBdiz/dense_1/kernel/RMSPropBdiz/dense_1/kernel/RMSProp_1Bdiz/dense_2/biasBdiz/dense_2/bias/RMSPropBdiz/dense_2/bias/RMSProp_1Bdiz/dense_2/kernelBdiz/dense_2/kernel/RMSPropBdiz/dense_2/kernel/RMSProp_1Bdiz/dense_3/biasBdiz/dense_3/bias/RMSPropBdiz/dense_3/bias/RMSProp_1Bdiz/dense_3/kernelBdiz/dense_3/kernel/RMSPropBdiz/dense_3/kernel/RMSProp_1Bencoder/conv2d/biasBencoder/conv2d/bias/AdamBencoder/conv2d/bias/Adam_1Bencoder/conv2d/kernelBencoder/conv2d/kernel/AdamBencoder/conv2d/kernel/Adam_1Bencoder/conv2d_1/biasBencoder/conv2d_1/bias/AdamBencoder/conv2d_1/bias/Adam_1Bencoder/conv2d_1/kernelBencoder/conv2d_1/kernel/AdamBencoder/conv2d_1/kernel/Adam_1Bencoder/dense/biasBencoder/dense/bias/AdamBencoder/dense/bias/Adam_1Bencoder/dense/kernelBencoder/dense/kernel/AdamBencoder/dense/kernel/Adam_1Bencoder/dense_1/biasBencoder/dense_1/bias/AdamBencoder/dense_1/bias/Adam_1Bencoder/dense_1/kernelBencoder/dense_1/kernel/AdamBencoder/dense_1/kernel/Adam_1Bgen/dense/biasBgen/dense/bias/RMSPropBgen/dense/bias/RMSProp_1Bgen/dense/kernelBgen/dense/kernel/RMSPropBgen/dense/kernel/RMSProp_1Bgen/dense_1/biasBgen/dense_1/bias/RMSPropBgen/dense_1/bias/RMSProp_1Bgen/dense_1/kernelBgen/dense_1/kernel/RMSPropBgen/dense_1/kernel/RMSProp_1Bgen/dense_2/biasBgen/dense_2/bias/RMSPropBgen/dense_2/bias/RMSProp_1Bgen/dense_2/kernelBgen/dense_2/kernel/RMSPropBgen/dense_2/kernel/RMSProp_1Bgen/dense_3/biasBgen/dense_3/bias/RMSPropBgen/dense_3/bias/RMSProp_1Bgen/dense_3/kernelBgen/dense_3/kernel/RMSPropBgen/dense_3/kernel/RMSProp_1*
dtype0
?
save/RestoreV2/shape_and_slicesConst*
_output_shapes	
:?*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*?
dtypes?
?2?*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
?
save/AssignAssignbeta1_powersave/RestoreV2*
T0*
use_locking(*
validate_shape(*
_output_shapes
: *3
_class)
'%loc:@decoder/batch_normalization/beta
?
save/Assign_1Assignbeta1_power_1save/RestoreV2:1*#
_class
loc:@disc/conv2d/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
?
save/Assign_2Assignbeta2_powersave/RestoreV2:2*
T0*3
_class)
'%loc:@decoder/batch_normalization/beta*
use_locking(*
validate_shape(*
_output_shapes
: 
?
save/Assign_3Assignbeta2_power_1save/RestoreV2:3*
_output_shapes
: *#
_class
loc:@disc/conv2d/bias*
validate_shape(*
T0*
use_locking(
?
save/Assign_4Assign decoder/batch_normalization/betasave/RestoreV2:4*
T0*3
_class)
'%loc:@decoder/batch_normalization/beta*
use_locking(*
_output_shapes	
:?*
validate_shape(
?
save/Assign_5Assign%decoder/batch_normalization/beta/Adamsave/RestoreV2:5*
_output_shapes	
:?*
use_locking(*
validate_shape(*
T0*3
_class)
'%loc:@decoder/batch_normalization/beta
?
save/Assign_6Assign'decoder/batch_normalization/beta/Adam_1save/RestoreV2:6*3
_class)
'%loc:@decoder/batch_normalization/beta*
T0*
use_locking(*
_output_shapes	
:?*
validate_shape(
?
save/Assign_7Assign!decoder/batch_normalization/gammasave/RestoreV2:7*4
_class*
(&loc:@decoder/batch_normalization/gamma*
use_locking(*
T0*
_output_shapes	
:?*
validate_shape(
?
save/Assign_8Assign&decoder/batch_normalization/gamma/Adamsave/RestoreV2:8*
validate_shape(*
_output_shapes	
:?*4
_class*
(&loc:@decoder/batch_normalization/gamma*
T0*
use_locking(
?
save/Assign_9Assign(decoder/batch_normalization/gamma/Adam_1save/RestoreV2:9*4
_class*
(&loc:@decoder/batch_normalization/gamma*
_output_shapes	
:?*
use_locking(*
validate_shape(*
T0
?
save/Assign_10Assign'decoder/batch_normalization/moving_meansave/RestoreV2:10*
validate_shape(*:
_class0
.,loc:@decoder/batch_normalization/moving_mean*
T0*
use_locking(*
_output_shapes	
:?
?
save/Assign_11Assign+decoder/batch_normalization/moving_variancesave/RestoreV2:11*
T0*>
_class4
20loc:@decoder/batch_normalization/moving_variance*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
save/Assign_12Assign"decoder/batch_normalization_1/betasave/RestoreV2:12*
_output_shapes	
:?*
use_locking(*
validate_shape(*
T0*5
_class+
)'loc:@decoder/batch_normalization_1/beta
?
save/Assign_13Assign'decoder/batch_normalization_1/beta/Adamsave/RestoreV2:13*
_output_shapes	
:?*
validate_shape(*
T0*5
_class+
)'loc:@decoder/batch_normalization_1/beta*
use_locking(
?
save/Assign_14Assign)decoder/batch_normalization_1/beta/Adam_1save/RestoreV2:14*5
_class+
)'loc:@decoder/batch_normalization_1/beta*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
save/Assign_15Assign#decoder/batch_normalization_1/gammasave/RestoreV2:15*6
_class,
*(loc:@decoder/batch_normalization_1/gamma*
use_locking(*
T0*
_output_shapes	
:?*
validate_shape(
?
save/Assign_16Assign(decoder/batch_normalization_1/gamma/Adamsave/RestoreV2:16*
T0*6
_class,
*(loc:@decoder/batch_normalization_1/gamma*
use_locking(*
_output_shapes	
:?*
validate_shape(
?
save/Assign_17Assign*decoder/batch_normalization_1/gamma/Adam_1save/RestoreV2:17*
_output_shapes	
:?*
T0*
validate_shape(*6
_class,
*(loc:@decoder/batch_normalization_1/gamma*
use_locking(
?
save/Assign_18Assign)decoder/batch_normalization_1/moving_meansave/RestoreV2:18*
use_locking(*
validate_shape(*<
_class2
0.loc:@decoder/batch_normalization_1/moving_mean*
_output_shapes	
:?*
T0
?
save/Assign_19Assign-decoder/batch_normalization_1/moving_variancesave/RestoreV2:19*
validate_shape(*
T0*@
_class6
42loc:@decoder/batch_normalization_1/moving_variance*
use_locking(*
_output_shapes	
:?
?
save/Assign_20Assigndecoder/layer_0/biassave/RestoreV2:20*
validate_shape(*
use_locking(*
_output_shapes	
:?*
T0*'
_class
loc:@decoder/layer_0/bias
?
save/Assign_21Assigndecoder/layer_0/bias/Adamsave/RestoreV2:21*
_output_shapes	
:?*
use_locking(*'
_class
loc:@decoder/layer_0/bias*
validate_shape(*
T0
?
save/Assign_22Assigndecoder/layer_0/bias/Adam_1save/RestoreV2:22*'
_class
loc:@decoder/layer_0/bias*
validate_shape(*
_output_shapes	
:?*
T0*
use_locking(
?
save/Assign_23Assigndecoder/layer_0/kernelsave/RestoreV2:23*
validate_shape(*
T0*
use_locking(*)
_class
loc:@decoder/layer_0/kernel*'
_output_shapes
:?d
?
save/Assign_24Assigndecoder/layer_0/kernel/Adamsave/RestoreV2:24*
T0*'
_output_shapes
:?d*
validate_shape(*
use_locking(*)
_class
loc:@decoder/layer_0/kernel
?
save/Assign_25Assigndecoder/layer_0/kernel/Adam_1save/RestoreV2:25*
use_locking(*)
_class
loc:@decoder/layer_0/kernel*
validate_shape(*'
_output_shapes
:?d*
T0
?
save/Assign_26Assigndecoder/layer_1/biassave/RestoreV2:26*
_output_shapes	
:?*
T0*
validate_shape(*'
_class
loc:@decoder/layer_1/bias*
use_locking(
?
save/Assign_27Assigndecoder/layer_1/bias/Adamsave/RestoreV2:27*
T0*
use_locking(*'
_class
loc:@decoder/layer_1/bias*
_output_shapes	
:?*
validate_shape(
?
save/Assign_28Assigndecoder/layer_1/bias/Adam_1save/RestoreV2:28*
T0*
_output_shapes	
:?*'
_class
loc:@decoder/layer_1/bias*
validate_shape(*
use_locking(
?
save/Assign_29Assigndecoder/layer_1/kernelsave/RestoreV2:29*
T0*
use_locking(*
validate_shape(*)
_class
loc:@decoder/layer_1/kernel*(
_output_shapes
:??
?
save/Assign_30Assigndecoder/layer_1/kernel/Adamsave/RestoreV2:30*(
_output_shapes
:??*)
_class
loc:@decoder/layer_1/kernel*
validate_shape(*
T0*
use_locking(
?
save/Assign_31Assigndecoder/layer_1/kernel/Adam_1save/RestoreV2:31*
use_locking(*
T0*)
_class
loc:@decoder/layer_1/kernel*(
_output_shapes
:??*
validate_shape(
?
save/Assign_32Assigndecoder/layer_2/biassave/RestoreV2:32*
_output_shapes
:*
validate_shape(*
T0*
use_locking(*'
_class
loc:@decoder/layer_2/bias
?
save/Assign_33Assigndecoder/layer_2/bias/Adamsave/RestoreV2:33*
use_locking(*
validate_shape(*
T0*'
_class
loc:@decoder/layer_2/bias*
_output_shapes
:
?
save/Assign_34Assigndecoder/layer_2/bias/Adam_1save/RestoreV2:34*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*'
_class
loc:@decoder/layer_2/bias
?
save/Assign_35Assigndecoder/layer_2/kernelsave/RestoreV2:35*
use_locking(*)
_class
loc:@decoder/layer_2/kernel*
T0*
validate_shape(*'
_output_shapes
:?
?
save/Assign_36Assigndecoder/layer_2/kernel/Adamsave/RestoreV2:36*
T0*
validate_shape(*)
_class
loc:@decoder/layer_2/kernel*
use_locking(*'
_output_shapes
:?
?
save/Assign_37Assigndecoder/layer_2/kernel/Adam_1save/RestoreV2:37*'
_output_shapes
:?*
validate_shape(*)
_class
loc:@decoder/layer_2/kernel*
T0*
use_locking(
?
save/Assign_38Assigndisc/conv2d/biassave/RestoreV2:38*#
_class
loc:@disc/conv2d/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
?
save/Assign_39Assigndisc/conv2d/bias/Adamsave/RestoreV2:39*
use_locking(*#
_class
loc:@disc/conv2d/bias*
T0*
validate_shape(*
_output_shapes
:
?
save/Assign_40Assigndisc/conv2d/bias/Adam_1save/RestoreV2:40*
_output_shapes
:*#
_class
loc:@disc/conv2d/bias*
T0*
use_locking(*
validate_shape(
?
save/Assign_41Assigndisc/conv2d/kernelsave/RestoreV2:41*
validate_shape(*
use_locking(*&
_output_shapes
:*%
_class
loc:@disc/conv2d/kernel*
T0
?
save/Assign_42Assigndisc/conv2d/kernel/Adamsave/RestoreV2:42*%
_class
loc:@disc/conv2d/kernel*
validate_shape(*&
_output_shapes
:*
T0*
use_locking(
?
save/Assign_43Assigndisc/conv2d/kernel/Adam_1save/RestoreV2:43*
use_locking(*
T0*%
_class
loc:@disc/conv2d/kernel*&
_output_shapes
:*
validate_shape(
?
save/Assign_44Assigndisc/conv2d_1/biassave/RestoreV2:44*%
_class
loc:@disc/conv2d_1/bias*
validate_shape(*
_output_shapes
:2*
use_locking(*
T0
?
save/Assign_45Assigndisc/conv2d_1/bias/Adamsave/RestoreV2:45*%
_class
loc:@disc/conv2d_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:2
?
save/Assign_46Assigndisc/conv2d_1/bias/Adam_1save/RestoreV2:46*
validate_shape(*
T0*
use_locking(*%
_class
loc:@disc/conv2d_1/bias*
_output_shapes
:2
?
save/Assign_47Assigndisc/conv2d_1/kernelsave/RestoreV2:47*&
_output_shapes
:2*
validate_shape(*
use_locking(*
T0*'
_class
loc:@disc/conv2d_1/kernel
?
save/Assign_48Assigndisc/conv2d_1/kernel/Adamsave/RestoreV2:48*
use_locking(*&
_output_shapes
:2*
T0*
validate_shape(*'
_class
loc:@disc/conv2d_1/kernel
?
save/Assign_49Assigndisc/conv2d_1/kernel/Adam_1save/RestoreV2:49*'
_class
loc:@disc/conv2d_1/kernel*
T0*&
_output_shapes
:2*
validate_shape(*
use_locking(
?
save/Assign_50Assigndisc/dense/biassave/RestoreV2:50*
_output_shapes	
:?*
T0*"
_class
loc:@disc/dense/bias*
validate_shape(*
use_locking(
?
save/Assign_51Assigndisc/dense/bias/Adamsave/RestoreV2:51*"
_class
loc:@disc/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
save/Assign_52Assigndisc/dense/bias/Adam_1save/RestoreV2:52*
validate_shape(*"
_class
loc:@disc/dense/bias*
_output_shapes	
:?*
use_locking(*
T0
?
save/Assign_53Assigndisc/dense/kernelsave/RestoreV2:53*
T0*
validate_shape(*$
_class
loc:@disc/dense/kernel*
use_locking(* 
_output_shapes
:
??
?
save/Assign_54Assigndisc/dense/kernel/Adamsave/RestoreV2:54*
T0*
validate_shape(*$
_class
loc:@disc/dense/kernel*
use_locking(* 
_output_shapes
:
??
?
save/Assign_55Assigndisc/dense/kernel/Adam_1save/RestoreV2:55*
validate_shape(*$
_class
loc:@disc/dense/kernel* 
_output_shapes
:
??*
use_locking(*
T0
?
save/Assign_56Assigndisc/dense_1/biassave/RestoreV2:56*
use_locking(*$
_class
loc:@disc/dense_1/bias*
validate_shape(*
T0*
_output_shapes
:
?
save/Assign_57Assigndisc/dense_1/bias/Adamsave/RestoreV2:57*
T0*$
_class
loc:@disc/dense_1/bias*
use_locking(*
_output_shapes
:*
validate_shape(
?
save/Assign_58Assigndisc/dense_1/bias/Adam_1save/RestoreV2:58*$
_class
loc:@disc/dense_1/bias*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
?
save/Assign_59Assigndisc/dense_1/kernelsave/RestoreV2:59*&
_class
loc:@disc/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	?*
T0
?
save/Assign_60Assigndisc/dense_1/kernel/Adamsave/RestoreV2:60*
validate_shape(*
T0*
_output_shapes
:	?*
use_locking(*&
_class
loc:@disc/dense_1/kernel
?
save/Assign_61Assigndisc/dense_1/kernel/Adam_1save/RestoreV2:61*
T0*
use_locking(*&
_class
loc:@disc/dense_1/kernel*
_output_shapes
:	?*
validate_shape(
?
save/Assign_62Assigndisc0/conv2d/biassave/RestoreV2:62*
_output_shapes
:*
validate_shape(*
use_locking(*
T0*$
_class
loc:@disc0/conv2d/bias
?
save/Assign_63Assigndisc0/conv2d/bias/RMSPropsave/RestoreV2:63*
_output_shapes
:*
T0*$
_class
loc:@disc0/conv2d/bias*
validate_shape(*
use_locking(
?
save/Assign_64Assigndisc0/conv2d/bias/RMSProp_1save/RestoreV2:64*
use_locking(*$
_class
loc:@disc0/conv2d/bias*
validate_shape(*
_output_shapes
:*
T0
?
save/Assign_65Assigndisc0/conv2d/kernelsave/RestoreV2:65*&
_class
loc:@disc0/conv2d/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
?
save/Assign_66Assigndisc0/conv2d/kernel/RMSPropsave/RestoreV2:66*
use_locking(*&
_class
loc:@disc0/conv2d/kernel*
validate_shape(*&
_output_shapes
:*
T0
?
save/Assign_67Assigndisc0/conv2d/kernel/RMSProp_1save/RestoreV2:67*
T0*&
_output_shapes
:*&
_class
loc:@disc0/conv2d/kernel*
use_locking(*
validate_shape(
?
save/Assign_68Assigndisc0/conv2d_1/biassave/RestoreV2:68*
use_locking(*
T0*
validate_shape(*
_output_shapes
:2*&
_class
loc:@disc0/conv2d_1/bias
?
save/Assign_69Assigndisc0/conv2d_1/bias/RMSPropsave/RestoreV2:69*&
_class
loc:@disc0/conv2d_1/bias*
T0*
_output_shapes
:2*
validate_shape(*
use_locking(
?
save/Assign_70Assigndisc0/conv2d_1/bias/RMSProp_1save/RestoreV2:70*&
_class
loc:@disc0/conv2d_1/bias*
_output_shapes
:2*
validate_shape(*
T0*
use_locking(
?
save/Assign_71Assigndisc0/conv2d_1/kernelsave/RestoreV2:71*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:2*(
_class
loc:@disc0/conv2d_1/kernel
?
save/Assign_72Assigndisc0/conv2d_1/kernel/RMSPropsave/RestoreV2:72*
T0*&
_output_shapes
:2*(
_class
loc:@disc0/conv2d_1/kernel*
use_locking(*
validate_shape(
?
save/Assign_73Assigndisc0/conv2d_1/kernel/RMSProp_1save/RestoreV2:73*&
_output_shapes
:2*
validate_shape(*
T0*
use_locking(*(
_class
loc:@disc0/conv2d_1/kernel
?
save/Assign_74Assigndisc0/dense/biassave/RestoreV2:74*
T0*
validate_shape(*#
_class
loc:@disc0/dense/bias*
use_locking(*
_output_shapes	
:?
?
save/Assign_75Assigndisc0/dense/bias/RMSPropsave/RestoreV2:75*
_output_shapes	
:?*
T0*
validate_shape(*#
_class
loc:@disc0/dense/bias*
use_locking(
?
save/Assign_76Assigndisc0/dense/bias/RMSProp_1save/RestoreV2:76*
use_locking(*
T0*
validate_shape(*#
_class
loc:@disc0/dense/bias*
_output_shapes	
:?
?
save/Assign_77Assigndisc0/dense/kernelsave/RestoreV2:77*%
_class
loc:@disc0/dense/kernel*
T0*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save/Assign_78Assigndisc0/dense/kernel/RMSPropsave/RestoreV2:78* 
_output_shapes
:
??*
use_locking(*
T0*
validate_shape(*%
_class
loc:@disc0/dense/kernel
?
save/Assign_79Assigndisc0/dense/kernel/RMSProp_1save/RestoreV2:79*
T0*%
_class
loc:@disc0/dense/kernel*
use_locking(*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_80Assigndisc0/dense_1/biassave/RestoreV2:80*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*%
_class
loc:@disc0/dense_1/bias
?
save/Assign_81Assigndisc0/dense_1/bias/RMSPropsave/RestoreV2:81*
validate_shape(*
T0*%
_class
loc:@disc0/dense_1/bias*
_output_shapes
:*
use_locking(
?
save/Assign_82Assigndisc0/dense_1/bias/RMSProp_1save/RestoreV2:82*
validate_shape(*%
_class
loc:@disc0/dense_1/bias*
use_locking(*
_output_shapes
:*
T0
?
save/Assign_83Assigndisc0/dense_1/kernelsave/RestoreV2:83*
use_locking(*
_output_shapes
:	?*
validate_shape(*
T0*'
_class
loc:@disc0/dense_1/kernel
?
save/Assign_84Assigndisc0/dense_1/kernel/RMSPropsave/RestoreV2:84*
T0*
_output_shapes
:	?*
validate_shape(*
use_locking(*'
_class
loc:@disc0/dense_1/kernel
?
save/Assign_85Assigndisc0/dense_1/kernel/RMSProp_1save/RestoreV2:85*'
_class
loc:@disc0/dense_1/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	?
?
save/Assign_86Assigndiz/dense/biassave/RestoreV2:86*
T0*
validate_shape(*!
_class
loc:@diz/dense/bias*
use_locking(*
_output_shapes
: 
?
save/Assign_87Assigndiz/dense/bias/RMSPropsave/RestoreV2:87*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*!
_class
loc:@diz/dense/bias
?
save/Assign_88Assigndiz/dense/bias/RMSProp_1save/RestoreV2:88*
validate_shape(*!
_class
loc:@diz/dense/bias*
use_locking(*
_output_shapes
: *
T0
?
save/Assign_89Assigndiz/dense/kernelsave/RestoreV2:89*#
_class
loc:@diz/dense/kernel*
use_locking(*
T0*
_output_shapes

:d *
validate_shape(
?
save/Assign_90Assigndiz/dense/kernel/RMSPropsave/RestoreV2:90*
validate_shape(*
use_locking(*#
_class
loc:@diz/dense/kernel*
T0*
_output_shapes

:d 
?
save/Assign_91Assigndiz/dense/kernel/RMSProp_1save/RestoreV2:91*
_output_shapes

:d *
validate_shape(*
T0*
use_locking(*#
_class
loc:@diz/dense/kernel
?
save/Assign_92Assigndiz/dense_1/biassave/RestoreV2:92*#
_class
loc:@diz/dense_1/bias*
_output_shapes
: *
use_locking(*
validate_shape(*
T0
?
save/Assign_93Assigndiz/dense_1/bias/RMSPropsave/RestoreV2:93*
_output_shapes
: *
T0*
validate_shape(*#
_class
loc:@diz/dense_1/bias*
use_locking(
?
save/Assign_94Assigndiz/dense_1/bias/RMSProp_1save/RestoreV2:94*
T0*
use_locking(*
validate_shape(*
_output_shapes
: *#
_class
loc:@diz/dense_1/bias
?
save/Assign_95Assigndiz/dense_1/kernelsave/RestoreV2:95*
T0*
use_locking(*
validate_shape(*
_output_shapes

:  *%
_class
loc:@diz/dense_1/kernel
?
save/Assign_96Assigndiz/dense_1/kernel/RMSPropsave/RestoreV2:96*
validate_shape(*%
_class
loc:@diz/dense_1/kernel*
use_locking(*
T0*
_output_shapes

:  
?
save/Assign_97Assigndiz/dense_1/kernel/RMSProp_1save/RestoreV2:97*
use_locking(*%
_class
loc:@diz/dense_1/kernel*
_output_shapes

:  *
validate_shape(*
T0
?
save/Assign_98Assigndiz/dense_2/biassave/RestoreV2:98*
_output_shapes
: *
T0*
use_locking(*#
_class
loc:@diz/dense_2/bias*
validate_shape(
?
save/Assign_99Assigndiz/dense_2/bias/RMSPropsave/RestoreV2:99*
use_locking(*#
_class
loc:@diz/dense_2/bias*
_output_shapes
: *
T0*
validate_shape(
?
save/Assign_100Assigndiz/dense_2/bias/RMSProp_1save/RestoreV2:100*
validate_shape(*
_output_shapes
: *
T0*
use_locking(*#
_class
loc:@diz/dense_2/bias
?
save/Assign_101Assigndiz/dense_2/kernelsave/RestoreV2:101*
use_locking(*
validate_shape(*
T0*%
_class
loc:@diz/dense_2/kernel*
_output_shapes

:  
?
save/Assign_102Assigndiz/dense_2/kernel/RMSPropsave/RestoreV2:102*
T0*%
_class
loc:@diz/dense_2/kernel*
_output_shapes

:  *
use_locking(*
validate_shape(
?
save/Assign_103Assigndiz/dense_2/kernel/RMSProp_1save/RestoreV2:103*
validate_shape(*
use_locking(*
_output_shapes

:  *%
_class
loc:@diz/dense_2/kernel*
T0
?
save/Assign_104Assigndiz/dense_3/biassave/RestoreV2:104*#
_class
loc:@diz/dense_3/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
?
save/Assign_105Assigndiz/dense_3/bias/RMSPropsave/RestoreV2:105*
_output_shapes
:*
use_locking(*
validate_shape(*#
_class
loc:@diz/dense_3/bias*
T0
?
save/Assign_106Assigndiz/dense_3/bias/RMSProp_1save/RestoreV2:106*
T0*#
_class
loc:@diz/dense_3/bias*
_output_shapes
:*
use_locking(*
validate_shape(
?
save/Assign_107Assigndiz/dense_3/kernelsave/RestoreV2:107*
validate_shape(*
use_locking(*%
_class
loc:@diz/dense_3/kernel*
T0*
_output_shapes

: 
?
save/Assign_108Assigndiz/dense_3/kernel/RMSPropsave/RestoreV2:108*
use_locking(*%
_class
loc:@diz/dense_3/kernel*
T0*
_output_shapes

: *
validate_shape(
?
save/Assign_109Assigndiz/dense_3/kernel/RMSProp_1save/RestoreV2:109*%
_class
loc:@diz/dense_3/kernel*
T0*
_output_shapes

: *
use_locking(*
validate_shape(
?
save/Assign_110Assignencoder/conv2d/biassave/RestoreV2:110*&
_class
loc:@encoder/conv2d/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
?
save/Assign_111Assignencoder/conv2d/bias/Adamsave/RestoreV2:111*
T0*&
_class
loc:@encoder/conv2d/bias*
validate_shape(*
use_locking(*
_output_shapes
:
?
save/Assign_112Assignencoder/conv2d/bias/Adam_1save/RestoreV2:112*&
_class
loc:@encoder/conv2d/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
?
save/Assign_113Assignencoder/conv2d/kernelsave/RestoreV2:113*
T0*
use_locking(*(
_class
loc:@encoder/conv2d/kernel*
validate_shape(*&
_output_shapes
:
?
save/Assign_114Assignencoder/conv2d/kernel/Adamsave/RestoreV2:114*&
_output_shapes
:*
T0*
validate_shape(*(
_class
loc:@encoder/conv2d/kernel*
use_locking(
?
save/Assign_115Assignencoder/conv2d/kernel/Adam_1save/RestoreV2:115*(
_class
loc:@encoder/conv2d/kernel*
T0*
use_locking(*&
_output_shapes
:*
validate_shape(
?
save/Assign_116Assignencoder/conv2d_1/biassave/RestoreV2:116*
validate_shape(*
use_locking(*(
_class
loc:@encoder/conv2d_1/bias*
T0*
_output_shapes
:2
?
save/Assign_117Assignencoder/conv2d_1/bias/Adamsave/RestoreV2:117*
_output_shapes
:2*
T0*
use_locking(*(
_class
loc:@encoder/conv2d_1/bias*
validate_shape(
?
save/Assign_118Assignencoder/conv2d_1/bias/Adam_1save/RestoreV2:118*
validate_shape(*
use_locking(*(
_class
loc:@encoder/conv2d_1/bias*
T0*
_output_shapes
:2
?
save/Assign_119Assignencoder/conv2d_1/kernelsave/RestoreV2:119*
T0*
use_locking(*
validate_shape(**
_class 
loc:@encoder/conv2d_1/kernel*&
_output_shapes
:2
?
save/Assign_120Assignencoder/conv2d_1/kernel/Adamsave/RestoreV2:120**
_class 
loc:@encoder/conv2d_1/kernel*
T0*
use_locking(*&
_output_shapes
:2*
validate_shape(
?
save/Assign_121Assignencoder/conv2d_1/kernel/Adam_1save/RestoreV2:121**
_class 
loc:@encoder/conv2d_1/kernel*
T0*&
_output_shapes
:2*
use_locking(*
validate_shape(
?
save/Assign_122Assignencoder/dense/biassave/RestoreV2:122*%
_class
loc:@encoder/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:d
?
save/Assign_123Assignencoder/dense/bias/Adamsave/RestoreV2:123*
use_locking(*
validate_shape(*
_output_shapes
:d*%
_class
loc:@encoder/dense/bias*
T0
?
save/Assign_124Assignencoder/dense/bias/Adam_1save/RestoreV2:124*%
_class
loc:@encoder/dense/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d
?
save/Assign_125Assignencoder/dense/kernelsave/RestoreV2:125*
validate_shape(*
_output_shapes
:	?d*
use_locking(*'
_class
loc:@encoder/dense/kernel*
T0
?
save/Assign_126Assignencoder/dense/kernel/Adamsave/RestoreV2:126*
_output_shapes
:	?d*'
_class
loc:@encoder/dense/kernel*
T0*
use_locking(*
validate_shape(
?
save/Assign_127Assignencoder/dense/kernel/Adam_1save/RestoreV2:127*
validate_shape(*
use_locking(*
_output_shapes
:	?d*'
_class
loc:@encoder/dense/kernel*
T0
?
save/Assign_128Assignencoder/dense_1/biassave/RestoreV2:128*'
_class
loc:@encoder/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:d*
T0
?
save/Assign_129Assignencoder/dense_1/bias/Adamsave/RestoreV2:129*'
_class
loc:@encoder/dense_1/bias*
_output_shapes
:d*
validate_shape(*
use_locking(*
T0
?
save/Assign_130Assignencoder/dense_1/bias/Adam_1save/RestoreV2:130*
use_locking(*'
_class
loc:@encoder/dense_1/bias*
_output_shapes
:d*
T0*
validate_shape(
?
save/Assign_131Assignencoder/dense_1/kernelsave/RestoreV2:131*)
_class
loc:@encoder/dense_1/kernel*
_output_shapes
:	?d*
use_locking(*
T0*
validate_shape(
?
save/Assign_132Assignencoder/dense_1/kernel/Adamsave/RestoreV2:132*
use_locking(*
T0*)
_class
loc:@encoder/dense_1/kernel*
_output_shapes
:	?d*
validate_shape(
?
save/Assign_133Assignencoder/dense_1/kernel/Adam_1save/RestoreV2:133*
use_locking(*)
_class
loc:@encoder/dense_1/kernel*
_output_shapes
:	?d*
validate_shape(*
T0
?
save/Assign_134Assigngen/dense/biassave/RestoreV2:134*
validate_shape(*
use_locking(*
T0*
_output_shapes
: *!
_class
loc:@gen/dense/bias
?
save/Assign_135Assigngen/dense/bias/RMSPropsave/RestoreV2:135*
use_locking(*
_output_shapes
: *
validate_shape(*!
_class
loc:@gen/dense/bias*
T0
?
save/Assign_136Assigngen/dense/bias/RMSProp_1save/RestoreV2:136*!
_class
loc:@gen/dense/bias*
validate_shape(*
_output_shapes
: *
T0*
use_locking(
?
save/Assign_137Assigngen/dense/kernelsave/RestoreV2:137*#
_class
loc:@gen/dense/kernel*
_output_shapes

:f *
use_locking(*
validate_shape(*
T0
?
save/Assign_138Assigngen/dense/kernel/RMSPropsave/RestoreV2:138*
_output_shapes

:f *
validate_shape(*
use_locking(*
T0*#
_class
loc:@gen/dense/kernel
?
save/Assign_139Assigngen/dense/kernel/RMSProp_1save/RestoreV2:139*
_output_shapes

:f *#
_class
loc:@gen/dense/kernel*
validate_shape(*
T0*
use_locking(
?
save/Assign_140Assigngen/dense_1/biassave/RestoreV2:140*
_output_shapes
: *
T0*
use_locking(*#
_class
loc:@gen/dense_1/bias*
validate_shape(
?
save/Assign_141Assigngen/dense_1/bias/RMSPropsave/RestoreV2:141*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*#
_class
loc:@gen/dense_1/bias
?
save/Assign_142Assigngen/dense_1/bias/RMSProp_1save/RestoreV2:142*
validate_shape(*
T0*
use_locking(*
_output_shapes
: *#
_class
loc:@gen/dense_1/bias
?
save/Assign_143Assigngen/dense_1/kernelsave/RestoreV2:143*
validate_shape(*
T0*
use_locking(*%
_class
loc:@gen/dense_1/kernel*
_output_shapes

:  
?
save/Assign_144Assigngen/dense_1/kernel/RMSPropsave/RestoreV2:144*
use_locking(*%
_class
loc:@gen/dense_1/kernel*
_output_shapes

:  *
T0*
validate_shape(
?
save/Assign_145Assigngen/dense_1/kernel/RMSProp_1save/RestoreV2:145*
T0*
validate_shape(*
use_locking(*%
_class
loc:@gen/dense_1/kernel*
_output_shapes

:  
?
save/Assign_146Assigngen/dense_2/biassave/RestoreV2:146*#
_class
loc:@gen/dense_2/bias*
use_locking(*
_output_shapes
: *
validate_shape(*
T0
?
save/Assign_147Assigngen/dense_2/bias/RMSPropsave/RestoreV2:147*
T0*
use_locking(*
validate_shape(*#
_class
loc:@gen/dense_2/bias*
_output_shapes
: 
?
save/Assign_148Assigngen/dense_2/bias/RMSProp_1save/RestoreV2:148*
use_locking(*
T0*
_output_shapes
: *
validate_shape(*#
_class
loc:@gen/dense_2/bias
?
save/Assign_149Assigngen/dense_2/kernelsave/RestoreV2:149*
validate_shape(*%
_class
loc:@gen/dense_2/kernel*
_output_shapes

:  *
use_locking(*
T0
?
save/Assign_150Assigngen/dense_2/kernel/RMSPropsave/RestoreV2:150*
T0*
validate_shape(*
_output_shapes

:  *%
_class
loc:@gen/dense_2/kernel*
use_locking(
?
save/Assign_151Assigngen/dense_2/kernel/RMSProp_1save/RestoreV2:151*
use_locking(*
validate_shape(*
T0*%
_class
loc:@gen/dense_2/kernel*
_output_shapes

:  
?
save/Assign_152Assigngen/dense_3/biassave/RestoreV2:152*
T0*
validate_shape(*
_output_shapes
:d*
use_locking(*#
_class
loc:@gen/dense_3/bias
?
save/Assign_153Assigngen/dense_3/bias/RMSPropsave/RestoreV2:153*
validate_shape(*#
_class
loc:@gen/dense_3/bias*
T0*
_output_shapes
:d*
use_locking(
?
save/Assign_154Assigngen/dense_3/bias/RMSProp_1save/RestoreV2:154*
validate_shape(*
T0*#
_class
loc:@gen/dense_3/bias*
use_locking(*
_output_shapes
:d
?
save/Assign_155Assigngen/dense_3/kernelsave/RestoreV2:155*%
_class
loc:@gen/dense_3/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

: d
?
save/Assign_156Assigngen/dense_3/kernel/RMSPropsave/RestoreV2:156*
T0*
validate_shape(*
_output_shapes

: d*
use_locking(*%
_class
loc:@gen/dense_3/kernel
?
save/Assign_157Assigngen/dense_3/kernel/RMSProp_1save/RestoreV2:157*
use_locking(*
T0*
_output_shapes

: d*
validate_shape(*%
_class
loc:@gen/dense_3/kernel
?
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_100^save/Assign_101^save/Assign_102^save/Assign_103^save/Assign_104^save/Assign_105^save/Assign_106^save/Assign_107^save/Assign_108^save/Assign_109^save/Assign_11^save/Assign_110^save/Assign_111^save/Assign_112^save/Assign_113^save/Assign_114^save/Assign_115^save/Assign_116^save/Assign_117^save/Assign_118^save/Assign_119^save/Assign_12^save/Assign_120^save/Assign_121^save/Assign_122^save/Assign_123^save/Assign_124^save/Assign_125^save/Assign_126^save/Assign_127^save/Assign_128^save/Assign_129^save/Assign_13^save/Assign_130^save/Assign_131^save/Assign_132^save/Assign_133^save/Assign_134^save/Assign_135^save/Assign_136^save/Assign_137^save/Assign_138^save/Assign_139^save/Assign_14^save/Assign_140^save/Assign_141^save/Assign_142^save/Assign_143^save/Assign_144^save/Assign_145^save/Assign_146^save/Assign_147^save/Assign_148^save/Assign_149^save/Assign_15^save/Assign_150^save/Assign_151^save/Assign_152^save/Assign_153^save/Assign_154^save/Assign_155^save/Assign_156^save/Assign_157^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_79^save/Assign_8^save/Assign_80^save/Assign_81^save/Assign_82^save/Assign_83^save/Assign_84^save/Assign_85^save/Assign_86^save/Assign_87^save/Assign_88^save/Assign_89^save/Assign_9^save/Assign_90^save/Assign_91^save/Assign_92^save/Assign_93^save/Assign_94^save/Assign_95^save/Assign_96^save/Assign_97^save/Assign_98^save/Assign_99
-
save/restore_allNoOp^save/restore_shard "&<
save/Const:0save/Identity:0save/restore_all (5 @F8";
train_op/
-
Adam
RMSProp
	RMSProp_1
	RMSProp_2
Adam_1"?

update_ops?
?
+decoder/batch_normalization/AssignMovingAvg
-decoder/batch_normalization/AssignMovingAvg_1
-decoder/batch_normalization_1/AssignMovingAvg
/decoder/batch_normalization_1/AssignMovingAvg_1
-decoder_1/batch_normalization/AssignMovingAvg
/decoder_1/batch_normalization/AssignMovingAvg_1
/decoder_1/batch_normalization_1/AssignMovingAvg
1decoder_1/batch_normalization_1/AssignMovingAvg_1"?
regularization_losses?
?
-diz/dense/kernel/Regularizer/l2_regularizer:0
+diz/dense/bias/Regularizer/l2_regularizer:0
/diz/dense_1/kernel/Regularizer/l2_regularizer:0
-diz/dense_1/bias/Regularizer/l2_regularizer:0
/diz/dense_2/kernel/Regularizer/l2_regularizer:0
-diz/dense_2/bias/Regularizer/l2_regularizer:0
/diz/dense_3/kernel/Regularizer/l2_regularizer:0
-diz/dense_3/bias/Regularizer/l2_regularizer:0
0disc0/conv2d/kernel/Regularizer/l2_regularizer:0
2disc0/conv2d_1/kernel/Regularizer/l2_regularizer:0
/disc0/dense/kernel/Regularizer/l2_regularizer:0
1disc0/dense_1/kernel/Regularizer/l2_regularizer:0
/disc/conv2d/kernel/Regularizer/l2_regularizer:0
1disc/conv2d_1/kernel/Regularizer/l2_regularizer:0
.disc/dense/kernel/Regularizer/l2_regularizer:0
0disc/dense_1/kernel/Regularizer/l2_regularizer:0"޵
	variablesϵ˵
?
encoder/conv2d/kernel:0encoder/conv2d/kernel/Assignencoder/conv2d/kernel/read:022encoder/conv2d/kernel/Initializer/random_uniform:08
z
encoder/conv2d/bias:0encoder/conv2d/bias/Assignencoder/conv2d/bias/read:02'encoder/conv2d/bias/Initializer/zeros:08
?
encoder/conv2d_1/kernel:0encoder/conv2d_1/kernel/Assignencoder/conv2d_1/kernel/read:024encoder/conv2d_1/kernel/Initializer/random_uniform:08
?
encoder/conv2d_1/bias:0encoder/conv2d_1/bias/Assignencoder/conv2d_1/bias/read:02)encoder/conv2d_1/bias/Initializer/zeros:08
?
encoder/dense/kernel:0encoder/dense/kernel/Assignencoder/dense/kernel/read:021encoder/dense/kernel/Initializer/random_uniform:08
v
encoder/dense/bias:0encoder/dense/bias/Assignencoder/dense/bias/read:02&encoder/dense/bias/Initializer/zeros:08
?
encoder/dense_1/kernel:0encoder/dense_1/kernel/Assignencoder/dense_1/kernel/read:023encoder/dense_1/kernel/Initializer/random_uniform:08
~
encoder/dense_1/bias:0encoder/dense_1/bias/Assignencoder/dense_1/bias/read:02(encoder/dense_1/bias/Initializer/zeros:08
w
gen/dense/kernel:0gen/dense/kernel/Assigngen/dense/kernel/read:02-gen/dense/kernel/Initializer/random_uniform:08
f
gen/dense/bias:0gen/dense/bias/Assigngen/dense/bias/read:02"gen/dense/bias/Initializer/zeros:08

gen/dense_1/kernel:0gen/dense_1/kernel/Assigngen/dense_1/kernel/read:02/gen/dense_1/kernel/Initializer/random_uniform:08
n
gen/dense_1/bias:0gen/dense_1/bias/Assigngen/dense_1/bias/read:02$gen/dense_1/bias/Initializer/zeros:08

gen/dense_2/kernel:0gen/dense_2/kernel/Assigngen/dense_2/kernel/read:02/gen/dense_2/kernel/Initializer/random_uniform:08
n
gen/dense_2/bias:0gen/dense_2/bias/Assigngen/dense_2/bias/read:02$gen/dense_2/bias/Initializer/zeros:08

gen/dense_3/kernel:0gen/dense_3/kernel/Assigngen/dense_3/kernel/read:02/gen/dense_3/kernel/Initializer/random_uniform:08
n
gen/dense_3/bias:0gen/dense_3/bias/Assigngen/dense_3/bias/read:02$gen/dense_3/bias/Initializer/zeros:08
w
diz/dense/kernel:0diz/dense/kernel/Assigndiz/dense/kernel/read:02-diz/dense/kernel/Initializer/random_uniform:08
f
diz/dense/bias:0diz/dense/bias/Assigndiz/dense/bias/read:02"diz/dense/bias/Initializer/zeros:08

diz/dense_1/kernel:0diz/dense_1/kernel/Assigndiz/dense_1/kernel/read:02/diz/dense_1/kernel/Initializer/random_uniform:08
n
diz/dense_1/bias:0diz/dense_1/bias/Assigndiz/dense_1/bias/read:02$diz/dense_1/bias/Initializer/zeros:08

diz/dense_2/kernel:0diz/dense_2/kernel/Assigndiz/dense_2/kernel/read:02/diz/dense_2/kernel/Initializer/random_uniform:08
n
diz/dense_2/bias:0diz/dense_2/bias/Assigndiz/dense_2/bias/read:02$diz/dense_2/bias/Initializer/zeros:08

diz/dense_3/kernel:0diz/dense_3/kernel/Assigndiz/dense_3/kernel/read:02/diz/dense_3/kernel/Initializer/random_uniform:08
n
diz/dense_3/bias:0diz/dense_3/bias/Assigndiz/dense_3/bias/read:02$diz/dense_3/bias/Initializer/zeros:08
?
decoder/layer_0/kernel:0decoder/layer_0/kernel/Assigndecoder/layer_0/kernel/read:025decoder/layer_0/kernel/Initializer/truncated_normal:08
~
decoder/layer_0/bias:0decoder/layer_0/bias/Assigndecoder/layer_0/bias/read:02(decoder/layer_0/bias/Initializer/zeros:08
?
#decoder/batch_normalization/gamma:0(decoder/batch_normalization/gamma/Assign(decoder/batch_normalization/gamma/read:024decoder/batch_normalization/gamma/Initializer/ones:08
?
"decoder/batch_normalization/beta:0'decoder/batch_normalization/beta/Assign'decoder/batch_normalization/beta/read:024decoder/batch_normalization/beta/Initializer/zeros:08
?
)decoder/batch_normalization/moving_mean:0.decoder/batch_normalization/moving_mean/Assign.decoder/batch_normalization/moving_mean/read:02;decoder/batch_normalization/moving_mean/Initializer/zeros:0@H
?
-decoder/batch_normalization/moving_variance:02decoder/batch_normalization/moving_variance/Assign2decoder/batch_normalization/moving_variance/read:02>decoder/batch_normalization/moving_variance/Initializer/ones:0@H
?
decoder/layer_1/kernel:0decoder/layer_1/kernel/Assigndecoder/layer_1/kernel/read:025decoder/layer_1/kernel/Initializer/truncated_normal:08
~
decoder/layer_1/bias:0decoder/layer_1/bias/Assigndecoder/layer_1/bias/read:02(decoder/layer_1/bias/Initializer/zeros:08
?
%decoder/batch_normalization_1/gamma:0*decoder/batch_normalization_1/gamma/Assign*decoder/batch_normalization_1/gamma/read:026decoder/batch_normalization_1/gamma/Initializer/ones:08
?
$decoder/batch_normalization_1/beta:0)decoder/batch_normalization_1/beta/Assign)decoder/batch_normalization_1/beta/read:026decoder/batch_normalization_1/beta/Initializer/zeros:08
?
+decoder/batch_normalization_1/moving_mean:00decoder/batch_normalization_1/moving_mean/Assign0decoder/batch_normalization_1/moving_mean/read:02=decoder/batch_normalization_1/moving_mean/Initializer/zeros:0@H
?
/decoder/batch_normalization_1/moving_variance:04decoder/batch_normalization_1/moving_variance/Assign4decoder/batch_normalization_1/moving_variance/read:02@decoder/batch_normalization_1/moving_variance/Initializer/ones:0@H
?
decoder/layer_2/kernel:0decoder/layer_2/kernel/Assigndecoder/layer_2/kernel/read:025decoder/layer_2/kernel/Initializer/truncated_normal:08
~
decoder/layer_2/bias:0decoder/layer_2/bias/Assigndecoder/layer_2/bias/read:02(decoder/layer_2/bias/Initializer/zeros:08
?
disc0/conv2d/kernel:0disc0/conv2d/kernel/Assigndisc0/conv2d/kernel/read:022disc0/conv2d/kernel/Initializer/truncated_normal:08
r
disc0/conv2d/bias:0disc0/conv2d/bias/Assigndisc0/conv2d/bias/read:02%disc0/conv2d/bias/Initializer/zeros:08
?
disc0/conv2d_1/kernel:0disc0/conv2d_1/kernel/Assigndisc0/conv2d_1/kernel/read:024disc0/conv2d_1/kernel/Initializer/truncated_normal:08
z
disc0/conv2d_1/bias:0disc0/conv2d_1/bias/Assigndisc0/conv2d_1/bias/read:02'disc0/conv2d_1/bias/Initializer/zeros:08
?
disc0/dense/kernel:0disc0/dense/kernel/Assigndisc0/dense/kernel/read:021disc0/dense/kernel/Initializer/truncated_normal:08
n
disc0/dense/bias:0disc0/dense/bias/Assigndisc0/dense/bias/read:02$disc0/dense/bias/Initializer/zeros:08
?
disc0/dense_1/kernel:0disc0/dense_1/kernel/Assigndisc0/dense_1/kernel/read:023disc0/dense_1/kernel/Initializer/truncated_normal:08
v
disc0/dense_1/bias:0disc0/dense_1/bias/Assigndisc0/dense_1/bias/read:02&disc0/dense_1/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
?
encoder/conv2d/kernel/Adam:0!encoder/conv2d/kernel/Adam/Assign!encoder/conv2d/kernel/Adam/read:02.encoder/conv2d/kernel/Adam/Initializer/zeros:0
?
encoder/conv2d/kernel/Adam_1:0#encoder/conv2d/kernel/Adam_1/Assign#encoder/conv2d/kernel/Adam_1/read:020encoder/conv2d/kernel/Adam_1/Initializer/zeros:0
?
encoder/conv2d/bias/Adam:0encoder/conv2d/bias/Adam/Assignencoder/conv2d/bias/Adam/read:02,encoder/conv2d/bias/Adam/Initializer/zeros:0
?
encoder/conv2d/bias/Adam_1:0!encoder/conv2d/bias/Adam_1/Assign!encoder/conv2d/bias/Adam_1/read:02.encoder/conv2d/bias/Adam_1/Initializer/zeros:0
?
encoder/conv2d_1/kernel/Adam:0#encoder/conv2d_1/kernel/Adam/Assign#encoder/conv2d_1/kernel/Adam/read:020encoder/conv2d_1/kernel/Adam/Initializer/zeros:0
?
 encoder/conv2d_1/kernel/Adam_1:0%encoder/conv2d_1/kernel/Adam_1/Assign%encoder/conv2d_1/kernel/Adam_1/read:022encoder/conv2d_1/kernel/Adam_1/Initializer/zeros:0
?
encoder/conv2d_1/bias/Adam:0!encoder/conv2d_1/bias/Adam/Assign!encoder/conv2d_1/bias/Adam/read:02.encoder/conv2d_1/bias/Adam/Initializer/zeros:0
?
encoder/conv2d_1/bias/Adam_1:0#encoder/conv2d_1/bias/Adam_1/Assign#encoder/conv2d_1/bias/Adam_1/read:020encoder/conv2d_1/bias/Adam_1/Initializer/zeros:0
?
encoder/dense/kernel/Adam:0 encoder/dense/kernel/Adam/Assign encoder/dense/kernel/Adam/read:02-encoder/dense/kernel/Adam/Initializer/zeros:0
?
encoder/dense/kernel/Adam_1:0"encoder/dense/kernel/Adam_1/Assign"encoder/dense/kernel/Adam_1/read:02/encoder/dense/kernel/Adam_1/Initializer/zeros:0
?
encoder/dense/bias/Adam:0encoder/dense/bias/Adam/Assignencoder/dense/bias/Adam/read:02+encoder/dense/bias/Adam/Initializer/zeros:0
?
encoder/dense/bias/Adam_1:0 encoder/dense/bias/Adam_1/Assign encoder/dense/bias/Adam_1/read:02-encoder/dense/bias/Adam_1/Initializer/zeros:0
?
encoder/dense_1/kernel/Adam:0"encoder/dense_1/kernel/Adam/Assign"encoder/dense_1/kernel/Adam/read:02/encoder/dense_1/kernel/Adam/Initializer/zeros:0
?
encoder/dense_1/kernel/Adam_1:0$encoder/dense_1/kernel/Adam_1/Assign$encoder/dense_1/kernel/Adam_1/read:021encoder/dense_1/kernel/Adam_1/Initializer/zeros:0
?
encoder/dense_1/bias/Adam:0 encoder/dense_1/bias/Adam/Assign encoder/dense_1/bias/Adam/read:02-encoder/dense_1/bias/Adam/Initializer/zeros:0
?
encoder/dense_1/bias/Adam_1:0"encoder/dense_1/bias/Adam_1/Assign"encoder/dense_1/bias/Adam_1/read:02/encoder/dense_1/bias/Adam_1/Initializer/zeros:0
?
decoder/layer_0/kernel/Adam:0"decoder/layer_0/kernel/Adam/Assign"decoder/layer_0/kernel/Adam/read:02/decoder/layer_0/kernel/Adam/Initializer/zeros:0
?
decoder/layer_0/kernel/Adam_1:0$decoder/layer_0/kernel/Adam_1/Assign$decoder/layer_0/kernel/Adam_1/read:021decoder/layer_0/kernel/Adam_1/Initializer/zeros:0
?
decoder/layer_0/bias/Adam:0 decoder/layer_0/bias/Adam/Assign decoder/layer_0/bias/Adam/read:02-decoder/layer_0/bias/Adam/Initializer/zeros:0
?
decoder/layer_0/bias/Adam_1:0"decoder/layer_0/bias/Adam_1/Assign"decoder/layer_0/bias/Adam_1/read:02/decoder/layer_0/bias/Adam_1/Initializer/zeros:0
?
(decoder/batch_normalization/gamma/Adam:0-decoder/batch_normalization/gamma/Adam/Assign-decoder/batch_normalization/gamma/Adam/read:02:decoder/batch_normalization/gamma/Adam/Initializer/zeros:0
?
*decoder/batch_normalization/gamma/Adam_1:0/decoder/batch_normalization/gamma/Adam_1/Assign/decoder/batch_normalization/gamma/Adam_1/read:02<decoder/batch_normalization/gamma/Adam_1/Initializer/zeros:0
?
'decoder/batch_normalization/beta/Adam:0,decoder/batch_normalization/beta/Adam/Assign,decoder/batch_normalization/beta/Adam/read:029decoder/batch_normalization/beta/Adam/Initializer/zeros:0
?
)decoder/batch_normalization/beta/Adam_1:0.decoder/batch_normalization/beta/Adam_1/Assign.decoder/batch_normalization/beta/Adam_1/read:02;decoder/batch_normalization/beta/Adam_1/Initializer/zeros:0
?
decoder/layer_1/kernel/Adam:0"decoder/layer_1/kernel/Adam/Assign"decoder/layer_1/kernel/Adam/read:02/decoder/layer_1/kernel/Adam/Initializer/zeros:0
?
decoder/layer_1/kernel/Adam_1:0$decoder/layer_1/kernel/Adam_1/Assign$decoder/layer_1/kernel/Adam_1/read:021decoder/layer_1/kernel/Adam_1/Initializer/zeros:0
?
decoder/layer_1/bias/Adam:0 decoder/layer_1/bias/Adam/Assign decoder/layer_1/bias/Adam/read:02-decoder/layer_1/bias/Adam/Initializer/zeros:0
?
decoder/layer_1/bias/Adam_1:0"decoder/layer_1/bias/Adam_1/Assign"decoder/layer_1/bias/Adam_1/read:02/decoder/layer_1/bias/Adam_1/Initializer/zeros:0
?
*decoder/batch_normalization_1/gamma/Adam:0/decoder/batch_normalization_1/gamma/Adam/Assign/decoder/batch_normalization_1/gamma/Adam/read:02<decoder/batch_normalization_1/gamma/Adam/Initializer/zeros:0
?
,decoder/batch_normalization_1/gamma/Adam_1:01decoder/batch_normalization_1/gamma/Adam_1/Assign1decoder/batch_normalization_1/gamma/Adam_1/read:02>decoder/batch_normalization_1/gamma/Adam_1/Initializer/zeros:0
?
)decoder/batch_normalization_1/beta/Adam:0.decoder/batch_normalization_1/beta/Adam/Assign.decoder/batch_normalization_1/beta/Adam/read:02;decoder/batch_normalization_1/beta/Adam/Initializer/zeros:0
?
+decoder/batch_normalization_1/beta/Adam_1:00decoder/batch_normalization_1/beta/Adam_1/Assign0decoder/batch_normalization_1/beta/Adam_1/read:02=decoder/batch_normalization_1/beta/Adam_1/Initializer/zeros:0
?
decoder/layer_2/kernel/Adam:0"decoder/layer_2/kernel/Adam/Assign"decoder/layer_2/kernel/Adam/read:02/decoder/layer_2/kernel/Adam/Initializer/zeros:0
?
decoder/layer_2/kernel/Adam_1:0$decoder/layer_2/kernel/Adam_1/Assign$decoder/layer_2/kernel/Adam_1/read:021decoder/layer_2/kernel/Adam_1/Initializer/zeros:0
?
decoder/layer_2/bias/Adam:0 decoder/layer_2/bias/Adam/Assign decoder/layer_2/bias/Adam/read:02-decoder/layer_2/bias/Adam/Initializer/zeros:0
?
decoder/layer_2/bias/Adam_1:0"decoder/layer_2/bias/Adam_1/Assign"decoder/layer_2/bias/Adam_1/read:02/decoder/layer_2/bias/Adam_1/Initializer/zeros:0
?
diz/dense/kernel/RMSProp:0diz/dense/kernel/RMSProp/Assigndiz/dense/kernel/RMSProp/read:02+diz/dense/kernel/RMSProp/Initializer/ones:0
?
diz/dense/kernel/RMSProp_1:0!diz/dense/kernel/RMSProp_1/Assign!diz/dense/kernel/RMSProp_1/read:02.diz/dense/kernel/RMSProp_1/Initializer/zeros:0
?
diz/dense/bias/RMSProp:0diz/dense/bias/RMSProp/Assigndiz/dense/bias/RMSProp/read:02)diz/dense/bias/RMSProp/Initializer/ones:0
?
diz/dense/bias/RMSProp_1:0diz/dense/bias/RMSProp_1/Assigndiz/dense/bias/RMSProp_1/read:02,diz/dense/bias/RMSProp_1/Initializer/zeros:0
?
diz/dense_1/kernel/RMSProp:0!diz/dense_1/kernel/RMSProp/Assign!diz/dense_1/kernel/RMSProp/read:02-diz/dense_1/kernel/RMSProp/Initializer/ones:0
?
diz/dense_1/kernel/RMSProp_1:0#diz/dense_1/kernel/RMSProp_1/Assign#diz/dense_1/kernel/RMSProp_1/read:020diz/dense_1/kernel/RMSProp_1/Initializer/zeros:0
?
diz/dense_1/bias/RMSProp:0diz/dense_1/bias/RMSProp/Assigndiz/dense_1/bias/RMSProp/read:02+diz/dense_1/bias/RMSProp/Initializer/ones:0
?
diz/dense_1/bias/RMSProp_1:0!diz/dense_1/bias/RMSProp_1/Assign!diz/dense_1/bias/RMSProp_1/read:02.diz/dense_1/bias/RMSProp_1/Initializer/zeros:0
?
diz/dense_2/kernel/RMSProp:0!diz/dense_2/kernel/RMSProp/Assign!diz/dense_2/kernel/RMSProp/read:02-diz/dense_2/kernel/RMSProp/Initializer/ones:0
?
diz/dense_2/kernel/RMSProp_1:0#diz/dense_2/kernel/RMSProp_1/Assign#diz/dense_2/kernel/RMSProp_1/read:020diz/dense_2/kernel/RMSProp_1/Initializer/zeros:0
?
diz/dense_2/bias/RMSProp:0diz/dense_2/bias/RMSProp/Assigndiz/dense_2/bias/RMSProp/read:02+diz/dense_2/bias/RMSProp/Initializer/ones:0
?
diz/dense_2/bias/RMSProp_1:0!diz/dense_2/bias/RMSProp_1/Assign!diz/dense_2/bias/RMSProp_1/read:02.diz/dense_2/bias/RMSProp_1/Initializer/zeros:0
?
diz/dense_3/kernel/RMSProp:0!diz/dense_3/kernel/RMSProp/Assign!diz/dense_3/kernel/RMSProp/read:02-diz/dense_3/kernel/RMSProp/Initializer/ones:0
?
diz/dense_3/kernel/RMSProp_1:0#diz/dense_3/kernel/RMSProp_1/Assign#diz/dense_3/kernel/RMSProp_1/read:020diz/dense_3/kernel/RMSProp_1/Initializer/zeros:0
?
diz/dense_3/bias/RMSProp:0diz/dense_3/bias/RMSProp/Assigndiz/dense_3/bias/RMSProp/read:02+diz/dense_3/bias/RMSProp/Initializer/ones:0
?
diz/dense_3/bias/RMSProp_1:0!diz/dense_3/bias/RMSProp_1/Assign!diz/dense_3/bias/RMSProp_1/read:02.diz/dense_3/bias/RMSProp_1/Initializer/zeros:0
?
disc0/conv2d/kernel/RMSProp:0"disc0/conv2d/kernel/RMSProp/Assign"disc0/conv2d/kernel/RMSProp/read:02.disc0/conv2d/kernel/RMSProp/Initializer/ones:0
?
disc0/conv2d/kernel/RMSProp_1:0$disc0/conv2d/kernel/RMSProp_1/Assign$disc0/conv2d/kernel/RMSProp_1/read:021disc0/conv2d/kernel/RMSProp_1/Initializer/zeros:0
?
disc0/conv2d/bias/RMSProp:0 disc0/conv2d/bias/RMSProp/Assign disc0/conv2d/bias/RMSProp/read:02,disc0/conv2d/bias/RMSProp/Initializer/ones:0
?
disc0/conv2d/bias/RMSProp_1:0"disc0/conv2d/bias/RMSProp_1/Assign"disc0/conv2d/bias/RMSProp_1/read:02/disc0/conv2d/bias/RMSProp_1/Initializer/zeros:0
?
disc0/conv2d_1/kernel/RMSProp:0$disc0/conv2d_1/kernel/RMSProp/Assign$disc0/conv2d_1/kernel/RMSProp/read:020disc0/conv2d_1/kernel/RMSProp/Initializer/ones:0
?
!disc0/conv2d_1/kernel/RMSProp_1:0&disc0/conv2d_1/kernel/RMSProp_1/Assign&disc0/conv2d_1/kernel/RMSProp_1/read:023disc0/conv2d_1/kernel/RMSProp_1/Initializer/zeros:0
?
disc0/conv2d_1/bias/RMSProp:0"disc0/conv2d_1/bias/RMSProp/Assign"disc0/conv2d_1/bias/RMSProp/read:02.disc0/conv2d_1/bias/RMSProp/Initializer/ones:0
?
disc0/conv2d_1/bias/RMSProp_1:0$disc0/conv2d_1/bias/RMSProp_1/Assign$disc0/conv2d_1/bias/RMSProp_1/read:021disc0/conv2d_1/bias/RMSProp_1/Initializer/zeros:0
?
disc0/dense/kernel/RMSProp:0!disc0/dense/kernel/RMSProp/Assign!disc0/dense/kernel/RMSProp/read:02-disc0/dense/kernel/RMSProp/Initializer/ones:0
?
disc0/dense/kernel/RMSProp_1:0#disc0/dense/kernel/RMSProp_1/Assign#disc0/dense/kernel/RMSProp_1/read:020disc0/dense/kernel/RMSProp_1/Initializer/zeros:0
?
disc0/dense/bias/RMSProp:0disc0/dense/bias/RMSProp/Assigndisc0/dense/bias/RMSProp/read:02+disc0/dense/bias/RMSProp/Initializer/ones:0
?
disc0/dense/bias/RMSProp_1:0!disc0/dense/bias/RMSProp_1/Assign!disc0/dense/bias/RMSProp_1/read:02.disc0/dense/bias/RMSProp_1/Initializer/zeros:0
?
disc0/dense_1/kernel/RMSProp:0#disc0/dense_1/kernel/RMSProp/Assign#disc0/dense_1/kernel/RMSProp/read:02/disc0/dense_1/kernel/RMSProp/Initializer/ones:0
?
 disc0/dense_1/kernel/RMSProp_1:0%disc0/dense_1/kernel/RMSProp_1/Assign%disc0/dense_1/kernel/RMSProp_1/read:022disc0/dense_1/kernel/RMSProp_1/Initializer/zeros:0
?
disc0/dense_1/bias/RMSProp:0!disc0/dense_1/bias/RMSProp/Assign!disc0/dense_1/bias/RMSProp/read:02-disc0/dense_1/bias/RMSProp/Initializer/ones:0
?
disc0/dense_1/bias/RMSProp_1:0#disc0/dense_1/bias/RMSProp_1/Assign#disc0/dense_1/bias/RMSProp_1/read:020disc0/dense_1/bias/RMSProp_1/Initializer/zeros:0
?
gen/dense/kernel/RMSProp:0gen/dense/kernel/RMSProp/Assigngen/dense/kernel/RMSProp/read:02+gen/dense/kernel/RMSProp/Initializer/ones:0
?
gen/dense/kernel/RMSProp_1:0!gen/dense/kernel/RMSProp_1/Assign!gen/dense/kernel/RMSProp_1/read:02.gen/dense/kernel/RMSProp_1/Initializer/zeros:0
?
gen/dense/bias/RMSProp:0gen/dense/bias/RMSProp/Assigngen/dense/bias/RMSProp/read:02)gen/dense/bias/RMSProp/Initializer/ones:0
?
gen/dense/bias/RMSProp_1:0gen/dense/bias/RMSProp_1/Assigngen/dense/bias/RMSProp_1/read:02,gen/dense/bias/RMSProp_1/Initializer/zeros:0
?
gen/dense_1/kernel/RMSProp:0!gen/dense_1/kernel/RMSProp/Assign!gen/dense_1/kernel/RMSProp/read:02-gen/dense_1/kernel/RMSProp/Initializer/ones:0
?
gen/dense_1/kernel/RMSProp_1:0#gen/dense_1/kernel/RMSProp_1/Assign#gen/dense_1/kernel/RMSProp_1/read:020gen/dense_1/kernel/RMSProp_1/Initializer/zeros:0
?
gen/dense_1/bias/RMSProp:0gen/dense_1/bias/RMSProp/Assigngen/dense_1/bias/RMSProp/read:02+gen/dense_1/bias/RMSProp/Initializer/ones:0
?
gen/dense_1/bias/RMSProp_1:0!gen/dense_1/bias/RMSProp_1/Assign!gen/dense_1/bias/RMSProp_1/read:02.gen/dense_1/bias/RMSProp_1/Initializer/zeros:0
?
gen/dense_2/kernel/RMSProp:0!gen/dense_2/kernel/RMSProp/Assign!gen/dense_2/kernel/RMSProp/read:02-gen/dense_2/kernel/RMSProp/Initializer/ones:0
?
gen/dense_2/kernel/RMSProp_1:0#gen/dense_2/kernel/RMSProp_1/Assign#gen/dense_2/kernel/RMSProp_1/read:020gen/dense_2/kernel/RMSProp_1/Initializer/zeros:0
?
gen/dense_2/bias/RMSProp:0gen/dense_2/bias/RMSProp/Assigngen/dense_2/bias/RMSProp/read:02+gen/dense_2/bias/RMSProp/Initializer/ones:0
?
gen/dense_2/bias/RMSProp_1:0!gen/dense_2/bias/RMSProp_1/Assign!gen/dense_2/bias/RMSProp_1/read:02.gen/dense_2/bias/RMSProp_1/Initializer/zeros:0
?
gen/dense_3/kernel/RMSProp:0!gen/dense_3/kernel/RMSProp/Assign!gen/dense_3/kernel/RMSProp/read:02-gen/dense_3/kernel/RMSProp/Initializer/ones:0
?
gen/dense_3/kernel/RMSProp_1:0#gen/dense_3/kernel/RMSProp_1/Assign#gen/dense_3/kernel/RMSProp_1/read:020gen/dense_3/kernel/RMSProp_1/Initializer/zeros:0
?
gen/dense_3/bias/RMSProp:0gen/dense_3/bias/RMSProp/Assigngen/dense_3/bias/RMSProp/read:02+gen/dense_3/bias/RMSProp/Initializer/ones:0
?
gen/dense_3/bias/RMSProp_1:0!gen/dense_3/bias/RMSProp_1/Assign!gen/dense_3/bias/RMSProp_1/read:02.gen/dense_3/bias/RMSProp_1/Initializer/zeros:0
?
disc/conv2d/kernel:0disc/conv2d/kernel/Assigndisc/conv2d/kernel/read:021disc/conv2d/kernel/Initializer/truncated_normal:08
n
disc/conv2d/bias:0disc/conv2d/bias/Assigndisc/conv2d/bias/read:02$disc/conv2d/bias/Initializer/zeros:08
?
disc/conv2d_1/kernel:0disc/conv2d_1/kernel/Assigndisc/conv2d_1/kernel/read:023disc/conv2d_1/kernel/Initializer/truncated_normal:08
v
disc/conv2d_1/bias:0disc/conv2d_1/bias/Assigndisc/conv2d_1/bias/read:02&disc/conv2d_1/bias/Initializer/zeros:08
}
disc/dense/kernel:0disc/dense/kernel/Assigndisc/dense/kernel/read:020disc/dense/kernel/Initializer/truncated_normal:08
j
disc/dense/bias:0disc/dense/bias/Assigndisc/dense/bias/read:02#disc/dense/bias/Initializer/zeros:08
?
disc/dense_1/kernel:0disc/dense_1/kernel/Assigndisc/dense_1/kernel/read:022disc/dense_1/kernel/Initializer/truncated_normal:08
r
disc/dense_1/bias:0disc/dense_1/bias/Assigndisc/dense_1/bias/read:02%disc/dense_1/bias/Initializer/zeros:08
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
?
disc/conv2d/kernel/Adam:0disc/conv2d/kernel/Adam/Assigndisc/conv2d/kernel/Adam/read:02+disc/conv2d/kernel/Adam/Initializer/zeros:0
?
disc/conv2d/kernel/Adam_1:0 disc/conv2d/kernel/Adam_1/Assign disc/conv2d/kernel/Adam_1/read:02-disc/conv2d/kernel/Adam_1/Initializer/zeros:0
?
disc/conv2d/bias/Adam:0disc/conv2d/bias/Adam/Assigndisc/conv2d/bias/Adam/read:02)disc/conv2d/bias/Adam/Initializer/zeros:0
?
disc/conv2d/bias/Adam_1:0disc/conv2d/bias/Adam_1/Assigndisc/conv2d/bias/Adam_1/read:02+disc/conv2d/bias/Adam_1/Initializer/zeros:0
?
disc/conv2d_1/kernel/Adam:0 disc/conv2d_1/kernel/Adam/Assign disc/conv2d_1/kernel/Adam/read:02-disc/conv2d_1/kernel/Adam/Initializer/zeros:0
?
disc/conv2d_1/kernel/Adam_1:0"disc/conv2d_1/kernel/Adam_1/Assign"disc/conv2d_1/kernel/Adam_1/read:02/disc/conv2d_1/kernel/Adam_1/Initializer/zeros:0
?
disc/conv2d_1/bias/Adam:0disc/conv2d_1/bias/Adam/Assigndisc/conv2d_1/bias/Adam/read:02+disc/conv2d_1/bias/Adam/Initializer/zeros:0
?
disc/conv2d_1/bias/Adam_1:0 disc/conv2d_1/bias/Adam_1/Assign disc/conv2d_1/bias/Adam_1/read:02-disc/conv2d_1/bias/Adam_1/Initializer/zeros:0
?
disc/dense/kernel/Adam:0disc/dense/kernel/Adam/Assigndisc/dense/kernel/Adam/read:02*disc/dense/kernel/Adam/Initializer/zeros:0
?
disc/dense/kernel/Adam_1:0disc/dense/kernel/Adam_1/Assigndisc/dense/kernel/Adam_1/read:02,disc/dense/kernel/Adam_1/Initializer/zeros:0
|
disc/dense/bias/Adam:0disc/dense/bias/Adam/Assigndisc/dense/bias/Adam/read:02(disc/dense/bias/Adam/Initializer/zeros:0
?
disc/dense/bias/Adam_1:0disc/dense/bias/Adam_1/Assigndisc/dense/bias/Adam_1/read:02*disc/dense/bias/Adam_1/Initializer/zeros:0
?
disc/dense_1/kernel/Adam:0disc/dense_1/kernel/Adam/Assigndisc/dense_1/kernel/Adam/read:02,disc/dense_1/kernel/Adam/Initializer/zeros:0
?
disc/dense_1/kernel/Adam_1:0!disc/dense_1/kernel/Adam_1/Assign!disc/dense_1/kernel/Adam_1/read:02.disc/dense_1/kernel/Adam_1/Initializer/zeros:0
?
disc/dense_1/bias/Adam:0disc/dense_1/bias/Adam/Assigndisc/dense_1/bias/Adam/read:02*disc/dense_1/bias/Adam/Initializer/zeros:0
?
disc/dense_1/bias/Adam_1:0disc/dense_1/bias/Adam_1/Assigndisc/dense_1/bias/Adam_1/read:02,disc/dense_1/bias/Adam_1/Initializer/zeros:0"?3
trainable_variables?3?3
?
encoder/conv2d/kernel:0encoder/conv2d/kernel/Assignencoder/conv2d/kernel/read:022encoder/conv2d/kernel/Initializer/random_uniform:08
z
encoder/conv2d/bias:0encoder/conv2d/bias/Assignencoder/conv2d/bias/read:02'encoder/conv2d/bias/Initializer/zeros:08
?
encoder/conv2d_1/kernel:0encoder/conv2d_1/kernel/Assignencoder/conv2d_1/kernel/read:024encoder/conv2d_1/kernel/Initializer/random_uniform:08
?
encoder/conv2d_1/bias:0encoder/conv2d_1/bias/Assignencoder/conv2d_1/bias/read:02)encoder/conv2d_1/bias/Initializer/zeros:08
?
encoder/dense/kernel:0encoder/dense/kernel/Assignencoder/dense/kernel/read:021encoder/dense/kernel/Initializer/random_uniform:08
v
encoder/dense/bias:0encoder/dense/bias/Assignencoder/dense/bias/read:02&encoder/dense/bias/Initializer/zeros:08
?
encoder/dense_1/kernel:0encoder/dense_1/kernel/Assignencoder/dense_1/kernel/read:023encoder/dense_1/kernel/Initializer/random_uniform:08
~
encoder/dense_1/bias:0encoder/dense_1/bias/Assignencoder/dense_1/bias/read:02(encoder/dense_1/bias/Initializer/zeros:08
w
gen/dense/kernel:0gen/dense/kernel/Assigngen/dense/kernel/read:02-gen/dense/kernel/Initializer/random_uniform:08
f
gen/dense/bias:0gen/dense/bias/Assigngen/dense/bias/read:02"gen/dense/bias/Initializer/zeros:08

gen/dense_1/kernel:0gen/dense_1/kernel/Assigngen/dense_1/kernel/read:02/gen/dense_1/kernel/Initializer/random_uniform:08
n
gen/dense_1/bias:0gen/dense_1/bias/Assigngen/dense_1/bias/read:02$gen/dense_1/bias/Initializer/zeros:08

gen/dense_2/kernel:0gen/dense_2/kernel/Assigngen/dense_2/kernel/read:02/gen/dense_2/kernel/Initializer/random_uniform:08
n
gen/dense_2/bias:0gen/dense_2/bias/Assigngen/dense_2/bias/read:02$gen/dense_2/bias/Initializer/zeros:08

gen/dense_3/kernel:0gen/dense_3/kernel/Assigngen/dense_3/kernel/read:02/gen/dense_3/kernel/Initializer/random_uniform:08
n
gen/dense_3/bias:0gen/dense_3/bias/Assigngen/dense_3/bias/read:02$gen/dense_3/bias/Initializer/zeros:08
w
diz/dense/kernel:0diz/dense/kernel/Assigndiz/dense/kernel/read:02-diz/dense/kernel/Initializer/random_uniform:08
f
diz/dense/bias:0diz/dense/bias/Assigndiz/dense/bias/read:02"diz/dense/bias/Initializer/zeros:08

diz/dense_1/kernel:0diz/dense_1/kernel/Assigndiz/dense_1/kernel/read:02/diz/dense_1/kernel/Initializer/random_uniform:08
n
diz/dense_1/bias:0diz/dense_1/bias/Assigndiz/dense_1/bias/read:02$diz/dense_1/bias/Initializer/zeros:08

diz/dense_2/kernel:0diz/dense_2/kernel/Assigndiz/dense_2/kernel/read:02/diz/dense_2/kernel/Initializer/random_uniform:08
n
diz/dense_2/bias:0diz/dense_2/bias/Assigndiz/dense_2/bias/read:02$diz/dense_2/bias/Initializer/zeros:08

diz/dense_3/kernel:0diz/dense_3/kernel/Assigndiz/dense_3/kernel/read:02/diz/dense_3/kernel/Initializer/random_uniform:08
n
diz/dense_3/bias:0diz/dense_3/bias/Assigndiz/dense_3/bias/read:02$diz/dense_3/bias/Initializer/zeros:08
?
decoder/layer_0/kernel:0decoder/layer_0/kernel/Assigndecoder/layer_0/kernel/read:025decoder/layer_0/kernel/Initializer/truncated_normal:08
~
decoder/layer_0/bias:0decoder/layer_0/bias/Assigndecoder/layer_0/bias/read:02(decoder/layer_0/bias/Initializer/zeros:08
?
#decoder/batch_normalization/gamma:0(decoder/batch_normalization/gamma/Assign(decoder/batch_normalization/gamma/read:024decoder/batch_normalization/gamma/Initializer/ones:08
?
"decoder/batch_normalization/beta:0'decoder/batch_normalization/beta/Assign'decoder/batch_normalization/beta/read:024decoder/batch_normalization/beta/Initializer/zeros:08
?
decoder/layer_1/kernel:0decoder/layer_1/kernel/Assigndecoder/layer_1/kernel/read:025decoder/layer_1/kernel/Initializer/truncated_normal:08
~
decoder/layer_1/bias:0decoder/layer_1/bias/Assigndecoder/layer_1/bias/read:02(decoder/layer_1/bias/Initializer/zeros:08
?
%decoder/batch_normalization_1/gamma:0*decoder/batch_normalization_1/gamma/Assign*decoder/batch_normalization_1/gamma/read:026decoder/batch_normalization_1/gamma/Initializer/ones:08
?
$decoder/batch_normalization_1/beta:0)decoder/batch_normalization_1/beta/Assign)decoder/batch_normalization_1/beta/read:026decoder/batch_normalization_1/beta/Initializer/zeros:08
?
decoder/layer_2/kernel:0decoder/layer_2/kernel/Assigndecoder/layer_2/kernel/read:025decoder/layer_2/kernel/Initializer/truncated_normal:08
~
decoder/layer_2/bias:0decoder/layer_2/bias/Assigndecoder/layer_2/bias/read:02(decoder/layer_2/bias/Initializer/zeros:08
?
disc0/conv2d/kernel:0disc0/conv2d/kernel/Assigndisc0/conv2d/kernel/read:022disc0/conv2d/kernel/Initializer/truncated_normal:08
r
disc0/conv2d/bias:0disc0/conv2d/bias/Assigndisc0/conv2d/bias/read:02%disc0/conv2d/bias/Initializer/zeros:08
?
disc0/conv2d_1/kernel:0disc0/conv2d_1/kernel/Assigndisc0/conv2d_1/kernel/read:024disc0/conv2d_1/kernel/Initializer/truncated_normal:08
z
disc0/conv2d_1/bias:0disc0/conv2d_1/bias/Assigndisc0/conv2d_1/bias/read:02'disc0/conv2d_1/bias/Initializer/zeros:08
?
disc0/dense/kernel:0disc0/dense/kernel/Assigndisc0/dense/kernel/read:021disc0/dense/kernel/Initializer/truncated_normal:08
n
disc0/dense/bias:0disc0/dense/bias/Assigndisc0/dense/bias/read:02$disc0/dense/bias/Initializer/zeros:08
?
disc0/dense_1/kernel:0disc0/dense_1/kernel/Assigndisc0/dense_1/kernel/read:023disc0/dense_1/kernel/Initializer/truncated_normal:08
v
disc0/dense_1/bias:0disc0/dense_1/bias/Assigndisc0/dense_1/bias/read:02&disc0/dense_1/bias/Initializer/zeros:08
?
disc/conv2d/kernel:0disc/conv2d/kernel/Assigndisc/conv2d/kernel/read:021disc/conv2d/kernel/Initializer/truncated_normal:08
n
disc/conv2d/bias:0disc/conv2d/bias/Assigndisc/conv2d/bias/read:02$disc/conv2d/bias/Initializer/zeros:08
?
disc/conv2d_1/kernel:0disc/conv2d_1/kernel/Assigndisc/conv2d_1/kernel/read:023disc/conv2d_1/kernel/Initializer/truncated_normal:08
v
disc/conv2d_1/bias:0disc/conv2d_1/bias/Assigndisc/conv2d_1/bias/read:02&disc/conv2d_1/bias/Initializer/zeros:08
}
disc/dense/kernel:0disc/dense/kernel/Assigndisc/dense/kernel/read:020disc/dense/kernel/Initializer/truncated_normal:08
j
disc/dense/bias:0disc/dense/bias/Assigndisc/dense/bias/read:02#disc/dense/bias/Initializer/zeros:08
?
disc/dense_1/kernel:0disc/dense_1/kernel/Assigndisc/dense_1/kernel/read:022disc/dense_1/kernel/Initializer/truncated_normal:08
r
disc/dense_1/bias:0disc/dense_1/bias/Assigndisc/dense_1/bias/read:02%disc/dense_1/bias/Initializer/zeros:08*?
serving_default?
 
X
X:0??????????

Y
Y:0?????????*
prob"
truediv_1:0?????????%
u 
	truediv:0?????????tensorflow/serving/predict