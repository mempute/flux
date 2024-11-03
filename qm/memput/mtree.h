#pragma once

#define nullx nullptr
#define T_LFT(p)	p->ptrLeft
#define T_RGT(p)	p->ptrRight

#define PTR_L(p)	T_LFT(p)
#define PTR_R(p)	T_RGT(p)
#define NEWL_L(p)	p->newLeft
#define NEWL_R(p)	p->newRight
#define PTR2_L(p)	p->ptrLeft2
#define PTR2_R(p)	p->ptrRight2
#define PTR3_L(p)	p->ptrLeft3
#define PTR3_R(p)	p->ptrRight3
#define PRO_L(p)	p->pPrevPro
#define PRO_R(p)	p->pNextPro
#define REQ_L(p)	p->pPrevReq
#define REQ_R(p)	p->pNextReq

#define EVAL(p) (*p)
#define P_LFT(m, p) m##_L(p)
#define P_RGT(m, p) m##_R(p)

#define ENDING_IT(m, n) if(n) P_RGT(m, P_LFT(m, n)) = nullx
#define CLOSE_IT(m, l, n) {\
	P_RGT(m, n) = nullx;\
	P_LFT(m, l) = n;\
}

//n의 right를 널로 마감하지않는다.
#define _REGIST_IT(m, l, n) {\
	if(l) {\
		P_RGT(m, P_LFT(m, l)) = n;\
		P_LFT(m, n) = P_LFT(m, l);\
		P_LFT(m, l) = n;\
	} else {\
		l = n;\
		P_LFT(m, n) = n;\
	}\
}
#define REGIST_IT(m, l, n) {\
	_REGIST_IT(m, l, n);\
	P_RGT(m, n) = nullx;\
}
//l list앞에 n list를 catination한다.
#define BEFORE_LIST(t, m, l, n) {\
	t *save;\
	if(l) {\
		save = P_LFT(m, n);\
		P_LFT(m, n) = P_LFT(m, l);\
		P_RGT(m, save) = l;\
		P_LFT(m, l) = save;\
	} else if(!P_RGT(m, n)) {\
		P_LFT(m, n) = n;\
		P_RGT(m, n) = nullx;\
	}\
	l = n;\
}
#define REGIST_LIST(t, m, l, n) {\
	t *save;\
	if(n) {\
		if(l) {\
			P_RGT(m, P_LFT(m, l)) = n;\
			save = P_LFT(m, n);\
			P_LFT(m, n) = P_LFT(m, l);\
			P_LFT(m, l) = save;\
			P_RGT(m, save) = nullx;\
		} else {\
			l = n;\
			if(!P_RGT(m, n)) {\
				P_LFT(m, n) = n;\
			}\
		}\
	}\
}
//n의 우측포인터를 널로 마감하지 않음
#define REGIST_LIST2(t, m, l, n) {\
	t *save;\
	if(n) {\
		if(l) {\
			P_RGT(m, P_LFT(m, l)) = n;\
			save = P_LFT(m, n);\
			P_LFT(m, n) = P_LFT(m, l);\
			P_LFT(m, l) = save;\
		} else {\
			l = n;\
			if(!P_RGT(m, n)) {\
				P_LFT(m, n) = n;\
			}\
		}\
	}\
}
//l의 앞에 n 삽입
#define HEAD_IT(m, l, n) {\
	if(l) {\
		P_LFT(m, n) = P_LFT(m, l);\
		P_RGT(m, n) = l;\
		P_LFT(m, l) = n;\
	} else {\
		P_LFT(m, n) = n;\
		P_RGT(m, n) = nullx;\
	}\
	l = n;\
}
//i앞에 n을 삽입한다.
#define BEFORE_IT(m, i, n) {\
	P_RGT(m, P_LFT(m, i)) = n;\
	P_LFT(m, n) = P_LFT(m, i);\
	P_RGT(m, n) = i;\
	P_LFT(m, i) = n;\
}
//i앞에 n을 삽입한다.
#define INSERT_IT(m, l, i, n) {\
	if(l == nullx || l == i) {\
		HEAD_IT(m, l, n);\
	} else {\
		BEFORE_IT(m, i, n);\
	}\
}
//i뒤에 n을 삽입한다.
#define AFTER_IT(m, i, n) {\
	P_LFT(m, P_RGT(m, i)) = n;\
	P_RGT(m, n) = P_RGT(m, i);\
	P_RGT(m, i) = n;\
	P_LFT(m, n) = i;\
}
#define DELETE_IT(m, r) {\
	P_RGT(m, P_LFT(m, r)) = P_RGT(m, r);\
	if(P_RGT(m, r)) P_LFT(m, P_RGT(m, r)) = P_LFT(m, r);\
}
#define REMOVE_IT(m, l, r) {\
	if(l == r) {\
		if(P_RGT(m, l)) {\
			P_LFT(m, P_RGT(m, l)) = P_LFT(m, l);\
			l = P_RGT(m, l);\
		} else l = nullx;\
	} else if(P_LFT(m, l) == r) {\
		P_LFT(m, l) = P_LFT(m, r);\
		P_RGT(m, P_LFT(m, r)) = 0;\
	} else {\
		DELETE_IT(m, r);\
	}\
}
//l 리스트로부터 d를 포함하여 d의 오른쪽 리스트를 분리한다.
#define DEVIDE_IT(t, m, l, d) {\
	if(l == d) l = nullx;\
	else {\
		t *last = P_LFT(m, l);\
		P_LFT(m, l) = P_LFT(m, d);\
		P_RGT(m, P_LFT(m, d)) = nullx;\
		P_LFT(m, d) = last;\
	}\
}
#define GET_IT(m, l, g) {\
	if(l) {\
		g = l;\
		if(P_RGT(m, l)) {\
			P_LFT(m, P_RGT(m, l)) = P_LFT(m, l);\
			l = P_RGT(m, l);\
		} else l = nullx;\
	} else g = nullx;\
}
//큐 운용 매크로
#define INSERT_TAIL(m, head, tail, n) {\
	if(tail) P_RGT(m, tail) = n;\
	else head = n;\
	P_RGT(m, n) = nullx;\
	tail = n;\
}
#define GET_HEAD(m, head, tail, n) {\
	n = head;\
	if(head) {\
		head = P_RGT(m, head);\
		if(head == nullx) tail = nullx;\
	}\
}

#define APPEND_SUBIT(m, p, l, n) {\
	REGIST_IT(m, l, n);\
	n->ptrParent = p;\
}
#define APPEND_CHILD(m, p, c) {\
	REGIST_IT(m, p->ptrChild, c);\
	c->ptrParent = p;\
}
#define FRONT_CHILD(m, p, c) {\
	HEAD_IT(m, p->ptrChild, c);\
	c->ptrParent = p;\
}

#define ENDING_LIST(n)			ENDING_IT(PTR, n)
#define CAT_LIST(t, l, n)		REGIST_LIST(t, PTR, l, n)
#define HEADCAT_LIST(t, l, n)	BEFORE_LIST(t, PTR, l, n)
#define JOIN_LIST(t, l, n)		REGIST_LIST2(t, PTR, l, n)
#define HEAD_LIST(l, n)			HEAD_IT(PTR, l, n)
#define _APPEND_LIST(l, n)		_REGIST_IT(PTR, l, n)
#define APPEND_LIST(l, n)		REGIST_IT(PTR, l, n)
#define FRONT_LIST(l, n)		HEAD_IT(PTR, l, n)
#define INSERT_BEFORE(i, n)		BEFORE_IT(PTR, i, n)
#define INSERT_LIST(l, i, n)	INSERT_IT(PTR, l, i, n)
#define INSERT_AFTER(i, n)		AFTER_IT(PTR, i, n)
#define CUT_LIST(l, r)			REMOVE_IT(PTR, l, r)
#define DELETE_LIST(r)			DELETE_IT(PTR, r)
#define GET_LIST(l, g)			GET_IT(PTR, l, g)
#define APART_LIST(t, l, r)		DEVIDE_IT(t, PTR, l, r)
#define APPEND_SLIST(p, l, n)	APPEND_SUBIT(PTR, p, l, n)
#define APPEND_CLIST(p, c)		APPEND_CHILD(PTR, p, c)
#define FRONT_CLIST(p, c)		FRONT_CHILD(PTR, p, c)
#define CLOSE_LIST(l, n)		CLOSE_IT(PTR, l, n)

#define CAT_NEWLIST(t, l, n)	REGIST_LIST(t, NEWL, l, n)
#define APPEND_NEWLIST(l, n)	REGIST_IT(NEWL, l, n)
#define CUT_NEWLIST(l, r)		REMOVE_IT(NEWL, l, r)
#define GET_NEWLIST(l, g)		GET_IT(NEWL, l, g)

#define ENDING_LIST2(n)			ENDING_IT(PTR2, n)
#define CAT_LIST2(t, l, n)	REGIST_LIST(t, PTR2, l, n)
#define _APPEND_LIST2(l, n)	_REGIST_IT(PTR2, l, n)
#define APPEND_LIST2(l, n)	REGIST_IT(PTR2, l, n)
#define FRONT_LIST2(l, n)		HEAD_IT(PTR2, l, n)
#define CUT_LIST2(l, r)		REMOVE_IT(PTR2, l, r)
#define GET_LIST2(l, g)		GET_IT(PTR2, l, g)
#define INSERT_LIST2(l, i, n) INSERT_IT(PTR2, l, i, n) //k48so.
#define APART_LIST2(t, l, r)	DEVIDE_IT(t, PTR2, l, r)
#define CLOSE_LIST2(l, n)		CLOSE_IT(PTR2, l, n)

#define CAT_LIST3(t, l, n)	REGIST_LIST(t, PTR3, l, n)
#define APPEND_LIST3(l, n)	REGIST_IT(PTR3, l, n)
#define FRONT_LIST3(l, n)		HEAD_IT(PTR3, l, n)
#define CUT_LIST3(l, r)		REMOVE_IT(PTR3, l, r)
#define GET_LIST3(l, g)		GET_IT(PTR3, l, g)
#define APART_LIST3(t, l, r)	DEVIDE_IT(t, PTR3, l, r)
//k239so.
#define APPEND_PROLIST(l, n)	REGIST_IT(PRO, l, n)
#define CUT_PROLIST(l, r)		REMOVE_IT(PRO, l, r)
#define INSERT_PROLIST(l, i, n)	INSERT_IT(PRO, l, i, n)
#define HEAD_PROLIST(l, n)		HEAD_IT(PRO, l, n)
#define INSERT_PROAFTER(i, n)	AFTER_IT(PRO, i, n)
#define CAT_PROLIST(t, l, n)	REGIST_LIST(t, PRO, l, n) //k332so.

#define APPEND_REQLIST(l, n)	REGIST_IT(REQ, l, n)
#define HEAD_REQLIST(l, n)		HEAD_IT(REQ, l, n)
#define GET_REQLIST(l, g)		GET_IT(REQ, l, g)

#define ApartListEndAppend(t, list_head, apart_head, apart_end) {\
	t *end_next;\
	if(apart_end->ptrRight) {\
		APART_LIST(t, list_head, apart_head);\
		end_next = apart_end->ptrRight;\
		APART_LIST(t, apart_head, end_next);\
		CAT_LIST(t, list_head, end_next);\
		CAT_LIST(t, list_head, apart_head);\
	}\
}

#define INSERT_QUEUE(head, tail, n)		INSERT_TAIL(PTR, head, tail, n)
#define GET_QUEUE(head, tail, n)		GET_HEAD(PTR, head, tail, n)

#define PDIC(t, p) (t *)p
#define VDIC(t, p)	*PDIC(t, p)

#define SL_APPEND(head, nd) {\
	nd->ptrRight = head;\
	head = nd;\
}
#define SL_GET(head, nd) {\
	nd = head;\
	head = head->ptrRight;\
}
#define TRAV_CMD_DOWN_LEFT	0
#define TRAV_CMD_DOWN_RIGHT	1
#define TRAV_CMD_UP			2
#define TRAV_STATUS_UP_LEFT	3
#define TRAV_CMD_NOT		4
//트리의 왼쪽으로만 끝까지 간다. 트리를 운행할때 처음 한번은 이 매크로을 호출해야함
#define LeftMostMTree(cur) {\
	for(;cur->ptrLeft; cur = cur->ptrLeft);\
}
#define LeftTopMostMTree(cur) {\
	for(;cur->ptrParent; cur = cur->ptrParent);\
	LeftMostMTree(cur);\
}
//위와 같으나 부모포인터를 연결한다, 처음 시작 cur의 ptrParent를 널로 리셋된 상태에서 시작해야 한다.
#define LLeftMostMTree(cur) {\
	for(;cur->ptrLeft; cur = cur->ptrLeft) cur->ptrLeft->ptrParent = cur;\
}
#define LLeftTopMostMTree(cur) {\
	for(;cur->ptrParent; cur = cur->ptrParent);\
	LLeftMostMTree(cur);\
}
#define LLeftTopMostMTree2(cur) {\
	cur->ptrParent = nullx;\
	LLeftMostMTree(cur);\
}
//널리스트를 TraverseRelation(TraverseMTree)할때 바로 끝나게하기위해 마지막의 부모를 널셋팅
#define SetEndTravMTree(trav, end, cur, sp) {\
	trav = cur->ptrLeft;\
	trav->ptrParent = nullx;\
	sp = nullx;\
	end = nullx;\
}
#define SetEndTravMTree2(locv, trav, end) {\
	trav = locv;\
	trav->ptrRight = nullx;\
	trav->ptrParent = nullx;\
	end = nullx;\
}

#define TraverseMTree(cur) {\
	if(cur->ptrRight) {\
		for(cur = cur->ptrRight;cur->ptrLeft; cur = cur->ptrLeft);\
	} else {\
		for(;cur->ptrParent; cur = cur->ptrParent) {\
			if(cur->ptrParent->ptrLeft == cur) break;/*cur가 상위의 우측이면 cur이하는 모두 탐색한 것이고 상위의 좌측이면 상위를 포함하여 그 우측이 앞으로 남은 노드*/\
		}\
		cur = cur->ptrParent;\
	}\
}
#define TraverseMTree2(cur, sib, next) {\
	if(sib) {\
		if(sib->ptrRight) sib = next = sib->ptrRight;\
		else {\
			sib = nullx;\
			TraverseMTree(cur);\
			next = cur;\
		}\
	} else {\
		if(cur->ptrSibling) sib = next = cur->ptrSibling;\
		else {\
			TraverseMTree(cur);\
			next = cur;\
		}\
	}\
}
//위와 같으나 parent link를 연결하면서 검색을 진행한다.
#define LTraverseMTree(cur) {\
	if(cur->ptrRight) {\
		cur->ptrRight->ptrParent = cur;\
		for(cur = cur->ptrRight;cur->ptrLeft; cur = cur->ptrLeft) cur->ptrLeft->ptrParent = cur;\
	} else {\
		for(;cur->ptrParent; cur = cur->ptrParent) {\
			if(cur->ptrParent->ptrLeft == cur) break;\
		}\
		cur = cur->ptrParent;\
	}\
}
#define LTraverseMTree2(cur, sib, next) {/*k514so.5.*/\
	if(sib) {\
		if(sib->ptrRight) sib = next = sib->ptrRight;\
		else {\
			sib = nullx;\
			LTraverseMTree(cur);\
			next = cur;\
		}\
	} else {\
		if(cur->ptrSibling) sib = next = cur->ptrSibling;\
		else {\
			LTraverseMTree(cur);\
			next = cur;\
		}\
	}\
}
//cur노드의 하나 뒤(다음) 노드를 리턴한다. cur노드의 우측이 있으면 우측의 최 좌측 노드가 다음 노드
//이고 그렇치않으면 cur노드가 그 상위의 좌측이면 그 상위노드가 하나 다음 노드이다.
#define NextMTreeNode(cur) {\
	if(cur->ptrRight) {\
		cur->ptrRight->ptrParent = cur;\
		cur = cur->ptrRight;\
		LeftMostMTree(cur);\
	} else {\
		for(;;) {\
			if(cur->ptrParent) {\
				if(cur->ptrParent->ptrRight == cur) cur = cur->ptrParent;\
				else { cur = cur->ptrParent; break; }\
			} else {\
				cur = nullx;\
				break;\
			}\
		}\
	}\
}