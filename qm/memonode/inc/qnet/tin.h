
#define SINGLE_REG_SIZE		8
#define DOUBLE_REG_SIZE		16
#define LTP_NONE			0
#define LTP_INT 			1	//single size value, intx load type
#define LTP_LONG 			2	//double size value, longx load type
#define LTP_DOUBLE 			3	//double size value, doublex load type
#define LTP_CLASS 			4	//double size value, class id load type

#define LoadFlag(sp) *sp
#define ResetFlag(sp) *(sp +1)
#define ThirdFlag(sp) *(sp +2)
#define ForthFlag(sp) *(sp +3)
#define IntValue_(sp) ((intx *)sp +1)
#define LongValue_(sp) ((longx *)sp +1)
#define DoubleValue_(sp) ((doublex *)sp +1)
#define IntValue(sp) *IntValue_(sp)
#define LongValue(sp) *LongValue_(sp)
#define DoubleValue(sp) *DoubleValue_(sp)
#define SizeClassValue(sp) IntValue(sp)
#define PointClassValue(sp) (bytex *)LongValue(sp)
#define MultiPartPoint(sp) ForthFlag(sp)
//lvalue <- rvalue
#define LoadIntValue(sp, v) {\
	LoadFlag(sp) = LTP_INT;\
	IntValue(sp) = v;\
}
#define LoadLongValue(sp, v) {\
	LoadFlag(sp) = LTP_LONG;\
	LongValue(sp) = v;\
}
#define LoadDoubleValue(sp, v) {\
	LoadFlag(sp) = LTP_DOUBLE;\
	DoubleValue(sp) = v;\
}
//호스트언어에서 MarshalPointArg로만 사용
#define LoadPointValue(sp, pv, sz, multi_part) {\
	LoadFlag(sp) = LTP_CLASS;\
	MultiPartPoint(sp) = multi_part;\
	SizeClassValue(sp) = sz;\
	LongValue(sp) = (longx)pv;\
}

typedef struct InstantPage_ {
	struct InstantPage_ *ptrLeft, *ptrRight;
} InstantPage;
