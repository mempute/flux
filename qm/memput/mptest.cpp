
#include "intracore.h"

void dot_bw_check(Flux *a, Flux *b, Flux *c, intx i)
{
	printf("-------------------------%d\n", i);
	c->dumpToGrad();
	a->resetGrad();
	b->resetGrad();
	c->backward();
	a->printg();
	b->printg();
	printf("\n\n");
}
void trs_bw_check(Flux *a, Flux *b)
{
	printf("-------------------------\n");
	b->dumpToGrad();
	a->resetGrad();
	b->backward();
	a->printg();
	printf("\n\n");
}
void kk(void)
{
	Tracer *tcr = trace(1);
	/*
	auto e = flux(tcr, { 2, 4, 3 }, tfloat, variable);
	e->arange(2 * 4 * 3)->printo();
	auto e2 = flux(tcr, { 2, 1, 3 }, tfloat, variable);
	e2->arange(2 * 1 * 3);
	auto e3 = flux(tcr, { 2, 2, 3 }, tfloat, variable);
	e3->arange(2 * 2 * 3);
	vector<Flux *> v;
	v.push_back(e);
	v.push_back(e2);
	v.push_back(e3);

	try {
		auto g = concat(&v, 1);
		g->printo();
	} catch(FaultObj er) {
		cout << er.fltmsg;
	}
	*/
	Flux *a, *b, *c;
	
	a = flux(tcr, { 2, 3 }, tfloat, variable);
	a->arange(2 * 3)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->dot(b, { {1}, {0} }, 0);
	c->printo();
	//dot_bw_check(a, b, c);
	
	a = flux(tcr, { 3,4 }, tfloat, variable);
	a->arange(3 * 4)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->dot(b, { {0}, {0} }, 0);
	c->printo();
	//dot_bw_check(a, b, c);

	a = flux(tcr, { 2,3 }, tfloat, variable);
	a->arange(2 * 3)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->dot(b, { {0}, {1} }, 0);
	c->printo();
	//dot_bw_check(a, b, c);
	
	a = flux(tcr, { 2,3 }, tfloat, variable);
	a->arange(2 * 3)->printo();
	b = flux(tcr, { 3,4,2 }, tfloat, variable);
	b->arange(3 * 4 * 2)->printo();
	c = a->dot(b, { {1}, {0} }, 0);
	c->printo();
	//dot_bw_check(a, b, c);
	
	a = flux(tcr, { 2,3,4 }, tfloat, variable);
	a->arange(2 * 3 * 4)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->dot(b, { {1}, {0} }, 0);
	c->printo();
	//dot_bw_check(a, b, c);

	a = flux(tcr, { 2,3 }, tfloat, variable);
	a->arange(2 * 3)->printo();
	b = flux(tcr, { 4,3,2 }, tfloat, variable);
	b->arange(4 * 3 * 2)->printo();
	c = a->dot(b, { {1}, {1} }, 0);
	c->printo();
	//dot_bw_check(a, b, c);

	a = flux(tcr, { 4,3,2 }, tfloat, variable);
	a->arange(4 * 3 * 2)->printo();
	b = flux(tcr, { 3,2,3 }, tfloat, variable);
	b->arange(3 * 2 * 3)->printo();
	c = a->dot(b, { {2}, {1} }, 0);
	c->printo();
	//dot_bw_check(a, b, c);

	a = flux(tcr, { 2,3,4,3 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 3)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->dot(b, { {1}, {0} }, 0);
	c->printo();
	//dot_bw_check(a, b, c);

	a = flux(tcr, { 4,3,2 }, tfloat, variable);
	a->arange(4 * 3 * 2)->printo();
	b = flux(tcr, { 2,3 }, tfloat, variable);
	b->arange(2 * 3)->printo();
	c = a->dot(b, { {2}, {0} }, 0);
	c->printo();
	//dot_bw_check(a, b, c);
	
	a = flux(tcr, { 4,3,2 }, tfloat, variable);
	a->arange(4 * 3 * 2)->printo();
	b = flux(tcr, { 3,2,3 }, tfloat, variable);
	b->arange(3 * 2 * 3)->printo();
	c = a->dot(b, { {1, 2}, {0, 1} }, 0);
	c->printo();//(0,1,2,3,4,5) * (0,3,6,9,12,15)
	//dot_bw_check(a, b, c);
	
	a = flux(tcr, { 4,3,2 }, tfloat, variable);
	a->arange(4 * 3 * 2)->printo();
	b = flux(tcr, { 3,2,3 }, tfloat, variable);
	b->arange(3 * 2 * 3)->printo();
	c = a->dot(b, { {1, 2}, {1, 0} }, 0);
	c->printo();//(0,1,2,3,4,5) * (0,6,12,3,9,15)
	//dot_bw_check(a, b, c);

	a = flux(tcr, { 4,3,2 }, tfloat, variable);
	a->arange(4 * 3 * 2)->printo();
	b = flux(tcr, { 3,2,2 }, tfloat, variable);
	b->arange(3 * 2 * 2)->printo();
	c = a->dot(b, { {1, 2}, {0, 2} }, 0);
	c->printo();//(0,1,2,3,4,5) * (0,1,4,5,8,9)
	//dot_bw_check(a, b, c);

	a = flux(tcr, { 4,2,6 }, tfloat, variable);
	a->arange(4 * 2 * 6)->printo();
	b = flux(tcr, { 3,4,3 }, tfloat, variable);
	b->arange(3 * 4 * 3)->printo();
	c = a->dot(b, { {1, 2}, {0, 1} }, 0);
	c->printo();
	//dot_bw_check(a, b, c);

	a = flux(tcr, { 2,3,4,2 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2)->printo();
	b = flux(tcr, { 2,3,4,2 }, tfloat, variable);
	b->arange(2 * 3 * 4 * 2)->printo();
	c = a->dot(b, { {1,3}, {1,3} }, 0);
	c->printo();
	//dot_bw_check(a, b, c);
	
	a = flux(tcr, { 2,3,4,2,4 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2 * 4)->printo();
	b = flux(tcr, { 2,3,4,2,4 }, tfloat, variable);
	b->arange(2 * 3 * 4 * 2 * 4)->printo();
	c = a->dot(b, { {1,3}, {1,3} }, 0);
	c->printo();
	//dot_bw_check(a, b, c);

	a = flux(tcr, { 2,3,4,2,1 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2 * 1)->printo();
	b = flux(tcr, { 2,3,4,2,1 }, tfloat, variable);
	b->arange(2 * 3 * 4 * 2 * 1)->printo();
	c = a->dot(b, { {2,3}, {2,3} }, 0);
	c->printo();
	//dot_bw_check(a, b, c);

	a = flux(tcr, { 2,3,4,2,1 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2 * 1)->printo();
	b = flux(tcr, { 2,3,4,2,1 }, tfloat, variable);
	b->arange(2 * 3 * 4 * 2 * 1)->printo();
	c = a->dot(b, { {1,4}, {1,4} }, 0);
	c->printo();
	//dot_bw_check(a, b, c);

	a = flux(tcr, { 4,3,2 }, tfloat, variable);
	a->arange(4 * 3 * 2)->printo();
	b = flux(tcr, { 6,2 }, tfloat, variable);
	b->arange(6 * 2)->printo();
	c = a->dot(b, { {1, 2}, {0} }, 0);
	c->printo();//(0,1,2,3,4,5) * (0,2,4,6,8,10)
	//dot_bw_check(a, b, c);

	a = flux(tcr, { 2,1,3,4 }, tfloat, variable);
	a->arange(2*1*3*4)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3*2)->printo();
	c = a->dot(b, { {2}, {0} }, 0);
	c->printo();
	
int i = 0;
	/*
	a = flux(tcr, { 2, 3 }, tfloat, variable);
	a->arange(2 * 3)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->dot(b, { {1}, {0} }, 0);
	//c->printo();
	dot_bw_check(a, b, c, i++);//0
	
	a = flux(tcr, { 3,4 }, tfloat, variable);
	a->arange(3 * 4)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->dot(b, { {0}, {0} }, 0);
	//c->printo();
	dot_bw_check(a, b, c, i++);//1
	
	a = flux(tcr, { 2,3 }, tfloat, variable);
	a->arange(2 * 3)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->dot(b, { {0}, {1} }, 0);
	//c->printo();
	dot_bw_check(a, b, c, i++);//2

	a = flux(tcr, { 2,3 }, tfloat, variable);
	a->arange(2 * 3)->printo();
	b = flux(tcr, { 3,4,2 }, tfloat, variable);
	b->arange(3 * 4 * 2)->printo();
	c = a->dot(b, { {1}, {0} }, 0);
	//c->printo();
	dot_bw_check(a, b, c, i++);//3
	
	a = flux(tcr, { 2,3,4 }, tfloat, variable);
	a->arange(2 * 3 * 4)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->dot(b, { {1}, {0} }, 0);
	//c->printo();
	dot_bw_check(a, b, c, i++);//4

	a = flux(tcr, { 2,3 }, tfloat, variable);
	a->arange(2 * 3)->printo();
	b = flux(tcr, { 4,3,2 }, tfloat, variable);
	b->arange(4 * 3 * 2)->printo();
	c = a->dot(b, { {1}, {1} }, 0);
	//c->printo();
	dot_bw_check(a, b, c, i++);//5

	a = flux(tcr, { 4,3,2 }, tfloat, variable);
	a->arange(4 * 3 * 2)->printo();
	b = flux(tcr, { 3,2,3 }, tfloat, variable);
	b->arange(3 * 2 * 3)->printo();
	c = a->dot(b, { {2}, {1} }, 0);
	//c->printo();
	dot_bw_check(a, b, c, i++);//6

	a = flux(tcr, { 2,3,4,3 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 3)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->dot(b, { {1}, {0} }, 0);
	//c->printo();
	dot_bw_check(a, b, c, i++);//7

	a = flux(tcr, { 4,3,2 }, tfloat, variable);
	a->arange(4 * 3 * 2)->printo();
	b = flux(tcr, { 2,3 }, tfloat, variable);
	b->arange(2 * 3)->printo();
	c = a->dot(b, { {2}, {0} }, 0);
	//c->printo();
	dot_bw_check(a, b, c, i++);//8

	a = flux(tcr, { 4,3,2 }, tfloat, variable);
	a->arange(4 * 3 * 2)->printo();
	b = flux(tcr, { 3,2,3 }, tfloat, variable);
	b->arange(3 * 2 * 3)->printo();
	c = a->dot(b, { {1, 2}, {0, 1} }, 0);
	//c->printo();//(0,1,2,3,4,5) * (0,3,6,9,12,15)
	dot_bw_check(a, b, c, i++);//9
	
	a = flux(tcr, { 4,3,2 }, tfloat, variable);
	a->arange(4 * 3 * 2)->printo();
	b = flux(tcr, { 3,2,3 }, tfloat, variable);
	b->arange(3 * 2 * 3)->printo();
	c = a->dot(b, { {1, 2}, {1, 0} }, 0);
	//c->printo();//(0,1,2,3,4,5) * (0,6,12,3,9,15)
	dot_bw_check(a, b, c, i++);//10
	
	a = flux(tcr, { 4,3,2 }, tfloat, variable);
	a->arange(4 * 3 * 2)->printo();
	b = flux(tcr, { 3,2,2 }, tfloat, variable);
	b->arange(3 * 2 * 2)->printo();
	c = a->dot(b, { {1, 2}, {0, 2} }, 0);
	//c->printo();//(0,1,2,3,4,5) * (0,1,4,5,8,9) 
	dot_bw_check(a, b, c, i++);//11

	a = flux(tcr, { 4,2,6 }, tfloat, variable);
	a->arange(4 * 2 * 6)->printo();
	b = flux(tcr, { 3,4,3 }, tfloat, variable);
	b->arange(3 * 4 * 3)->printo();
	c = a->dot(b, { {1, 2}, {0, 1} }, 0);
	//c->printo();
	dot_bw_check(a, b, c, i++);//12

	a = flux(tcr, { 2,3,4,2 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2)->printo();
	b = flux(tcr, { 2,3,4,2 }, tfloat, variable);
	b->arange(2 * 3 * 4 * 2)->printo();
	c = a->dot(b, { {1,3}, {1,3} }, 0);
	//c->printo();
	dot_bw_check(a, b, c, i++);//13
	
	a = flux(tcr, { 2,3,4,2,4 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2 * 4)->printo();
	b = flux(tcr, { 2,3,4,2,4 }, tfloat, variable);
	b->arange(2 * 3 * 4 * 2 * 4)->printo();
	c = a->dot(b, { {1,3}, {1,3} }, 0);
	//c->printo();
	dot_bw_check(a, b, c, i++);//14

	a = flux(tcr, { 2,3,4,2,1 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2 * 1)->printo();
	b = flux(tcr, { 2,3,4,2,1 }, tfloat, variable);
	b->arange(2 * 3 * 4 * 2 * 1)->printo();
	c = a->dot(b, { {2,3}, {2,3} }, 0);
	//c->printo();
	dot_bw_check(a, b, c, i++);//15

	a = flux(tcr, { 2,3,4,2,1 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2 * 1)->printo();
	b = flux(tcr, { 2,3,4,2,1 }, tfloat, variable);
	b->arange(2 * 3 * 4 * 2 * 1)->printo();
	c = a->dot(b, { {1,4}, {1,4} }, 0);
	//c->printo();
	dot_bw_check(a, b, c, i++);//16
	
	a = flux(tcr, { 4,3,2 }, tfloat, variable);
	a->arange(4 * 3 * 2)->printo();
	b = flux(tcr, { 6,2 }, tfloat, variable);
	b->arange(6 * 2)->printo();
	c = a->dot(b, { {1, 2}, {0} }, 0);
	//c->printo();//(0,1,2,3,4,5) * (0,2,4,6,8,10)
	dot_bw_check(a, b, c, i++);//17

	a = flux(tcr, { 2, 4, 3 }, tfloat, variable);
	a->arange(2 * 4 * 3)->printo();
	b = a->reshape({ -1, 2, 3 });
	b->printo();
	trs_bw_check(a, b);
	*/
	delete tcr;
}
void mul_bw_check(Flux *a, Flux *b, Flux *c) //ABP_BWTEST 옮겨야함.
{
	printf("-------------------------\n");
	c->dumpToGrad();
	a->resetGrad();
	b->resetGrad();
	c->backward();
	a->printg();
	b->printg();
	printf("\n\n");
}
void gg(void)
{
	Tracer *tcr = trace(1);
	
	Flux *a, *b, *c;

	//tcr->modeset(1);
	/*
	a = flux(tcr, { 3, 2 }, tfloat, variable);
	a->arange(3*2)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->mul(b);
	c->printo();

	a = flux(tcr, { 2, 3, 2 }, tfloat, variable);
	a->arange(2*3 * 2)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->mul(b);
	c->printo();

	a = flux(tcr, { 3, 2 }, tfloat, variable);
	a->arange(3 * 2)->printo();
	b = flux(tcr, { 2,3,2 }, tfloat, variable);
	b->arange(2*3 * 2)->printo();
	c = a->mul(b);
	c->printo();
	
	a = flux(tcr, { 2,1, 3, 2 }, tfloat, variable);
	a->arange(2 * 3 * 2)->printo();
	b = flux(tcr, { 3,3,2 }, tfloat, variable);
	b->arange(3*3 * 2)->printo();
	c = a->mul(b);
	c->printo();

	a = flux(tcr, { 2,1, 3, 2 }, tfloat, variable);
	a->arange(2 * 3 * 2)->printo();
	b = flux(tcr, { 2,3,3,2 }, tfloat, variable);
	b->arange(2 * 3 * 3 * 2)->printo();
	c = a->mul(b);
	c->printo();

	a = flux(tcr, { 2,1,1, 3, 2 }, tfloat, variable);
	a->arange(2 * 3 * 2)->printo();
	b = flux(tcr, { 2,3,2,3,2 }, tfloat, variable);
	b->arange(2 * 3 * 2* 3 * 2)->printo();
	c = a->mul(b);
	c->printo();

	a = flux(tcr, { 2,1,3,2 }, tfloat, variable);
	a->arange(2 * 3*2)->printo();
	b = flux(tcr, { 2,3,3,2 }, tfloat, variable);
	b->arange(2 * 3 * 3 * 2)->printo();
	c = a->div(b);
	c->printo();

	a = flux(tcr, { 2,1,3,2 }, tfloat, variable);
	a->arange(2 * 3 * 2)->printo();
	b = flux(tcr, { 3,3,2 }, tfloat, variable);
	b->arange(3 * 3 * 2)->printo();
	c = a->plus(b);
	c->printo();

	a = flux(tcr, { 2,1,1,1, 3, 2 }, tfloat, variable);
	a->arange(2 * 3 * 2)->printo();
	b = flux(tcr, { 2,3,2,2,3,2 }, tfloat, variable);
	b->arange(2 * 3 * 2 *2* 3 * 2)->printo();
	c = a->mul(b);
	c->printo();

	a = flux(tcr, { 2,1,1, 3, 2 }, tfloat, variable);
	a->arange(2 * 3 * 2)->printo();
	b = flux(tcr, { 3,3,2 }, tfloat, variable);
	b->arange(3*3 * 2)->printo();
	c = a->mul(b);
	c->printo();

	a = flux(tcr, { 2,4,1,2,1 }, tfloat, variable);
	a->arange(2 * 4 * 2)->printo();
	b = flux(tcr, { 3,2,1 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->mul(b);
	c->printo();

	a = flux(tcr, { 2,4,1,2,1,1 }, tfloat, variable);
	a->arange(2 * 4 * 2)->printo();
	b = flux(tcr, { 3,2,1,1 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->mul(b);
	c->printo();
	

	a = flux(tcr, { 2,4,1,2,1,1,2 }, tfloat, variable);
	a->arange(2 * 4 * 2*2)->printo();
	b = flux(tcr, { 3,2,1,1,2 }, tfloat, variable);
	b->arange(3 * 2*2)->printo();
	c = a->mul(b);
	c->printo();

	a = flux(tcr, { 2,1,2,1,1,2 }, tfloat, variable);
	a->arange(2*2 * 2)->printo();
	b = flux(tcr, { 3,2,1,1,2 }, tfloat, variable);
	b->arange(3 * 2*2)->printo();
	c = a->mul(b);
	c->printo();


	a = flux(tcr, { 2,1,2,2,1,2 }, tfloat, variable);
	a->arange(2 * 2 * 2 *2)->printo();
	b = flux(tcr, {   3,2,1,3,2 }, tfloat, variable);
	b->arange(3 * 2 * 3*2)->printo();
	c = a->mul(b);
	c->printo();
	
	a = flux(tcr, { 1,1, 3, 2 }, tfloat, variable);
	a->arange(3 * 2)->printo();
	b = flux(tcr, { 2,3,3,2 }, tfloat, variable);
	b->arange(2 * 3 * 3 * 2)->printo();
	c = a->mul(b);
	c->printo();

	a = flux(tcr, { 1,2, 3, 1 }, tfloat, variable);
	a->arange(3 * 2)->printo();
	b = flux(tcr, { 2,2,3,1 }, tfloat, variable);
	b->arange(2 * 2 * 3 * 1)->printo();
	c = a->mul(b);
	c->printo();
	*/
	/*
	a = flux(tcr, { 2,4,3 }, tfloat, variable);
	a->arange(-1)->printo();
	a = a->expand_dims(1);

	b = flux(tcr, { 5,4,3 }, tfloat, variable);
	b->fill(1.0);
	b->printo();

	c = *b * *a;
	c->printo();
	*/
	a = flux(tcr, { 3, 2 }, tfloat, variable);
	a->arange(3 * 2)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->mul(b);
	//c->printo();
	mul_bw_check(a, b, c);
	
	a = flux(tcr, { 2, 3, 2 }, tfloat, variable);
	a->arange(2 * 3 * 2)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->mul(b);
	//c->printo();
	mul_bw_check(a, b, c);

	a = flux(tcr, { 3, 2 }, tfloat, variable);
	a->arange(3 * 2)->printo();
	b = flux(tcr, { 2,3,2 }, tfloat, variable);
	b->arange(2 * 3 * 2)->printo();
	c = a->mul(b);
	//c->printo();
	mul_bw_check(a, b, c);
	
	a = flux(tcr, { 2,1, 3, 2 }, tfloat, variable);
	a->arange(2 * 3 * 2)->printo();
	b = flux(tcr, { 3,3,2 }, tfloat, variable);
	b->arange(3 * 3 * 2)->printo();
	c = a->mul(b);
	//c->printo();
	mul_bw_check(a, b, c);

	a = flux(tcr, { 2,1, 3, 2 }, tfloat, variable);
	a->arange(2 * 3 * 2)->printo();
	b = flux(tcr, { 2,3,3,2 }, tfloat, variable);
	b->arange(2 * 3 * 3 * 2)->printo();
	c = a->mul(b);
	//c->printo();
	mul_bw_check(a, b, c);

	a = flux(tcr, { 2,1,1, 3, 2 }, tfloat, variable);
	a->arange(2 * 3 * 2)->printo();
	b = flux(tcr, { 2,3,2,3,2 }, tfloat, variable);
	b->arange(2 * 3 * 2 * 3 * 2)->printo();
	c = a->mul(b);
	//c->printo();
	mul_bw_check(a, b, c);

	a = flux(tcr, { 2,1,1,1, 3, 2 }, tfloat, variable);
	a->arange(2 * 3 * 2)->printo();
	b = flux(tcr, { 2,3,2,2,3,2 }, tfloat, variable);
	b->arange(2 * 3 * 2 * 2 * 3 * 2)->printo();
	c = a->mul(b);
	//c->printo();
	mul_bw_check(a, b, c);

	a = flux(tcr, { 2,1,1, 3, 2 }, tfloat, variable);
	a->arange(2 * 3 * 2)->printo();
	b = flux(tcr, { 3,3,2 }, tfloat, variable);
	b->arange(3 * 3 * 2)->printo();
	c = a->mul(b);
	//c->printo();
	mul_bw_check(a, b, c);

	a = flux(tcr, { 2,4,1,2,1 }, tfloat, variable);
	a->arange(2 * 4 * 2)->printo();
	b = flux(tcr, { 3,2,1 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->mul(b);
	//c->printo();
	mul_bw_check(a, b, c);

	a = flux(tcr, { 2,4,1,2,1,1 }, tfloat, variable);
	a->arange(2 * 4 * 2)->printo();
	b = flux(tcr, { 3,2,1,1 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->mul(b);
	//c->printo();
	mul_bw_check(a, b, c);


	a = flux(tcr, { 2,4,1,2,1,1,2 }, tfloat, variable);
	a->arange(2 * 4 * 2 * 2)->printo();
	b = flux(tcr, { 3,2,1,1,2 }, tfloat, variable);
	b->arange(3 * 2 * 2)->printo();
	c = a->mul(b);
	//c->printo();
	mul_bw_check(a, b, c);

	a = flux(tcr, { 2,1,2,1,1,2 }, tfloat, variable);
	a->arange(2 * 2 * 2)->printo();
	b = flux(tcr, { 3,2,1,1,2 }, tfloat, variable);
	b->arange(3 * 2 * 2)->printo();
	c = a->mul(b);
	//c->printo();
	mul_bw_check(a, b, c);

	a = flux(tcr, { 2,1,2,2,1,2 }, tfloat, variable);
	a->arange(2 * 2 * 2 * 2)->printo();
	b = flux(tcr, { 3,2,1,3,2 }, tfloat, variable);
	b->arange(3 * 2 * 3 * 2)->printo();
	c = a->mul(b);
	//c->printo();
	mul_bw_check(a, b, c);

	a = flux(tcr, { 1,1, 3, 2 }, tfloat, variable);
	a->arange( 3 * 2)->printo();
	b = flux(tcr, { 2,3,3,2 }, tfloat, variable);
	b->arange(2 * 3 * 3 * 2)->printo();
	c = a->mul(b);
	//c->printo();
	mul_bw_check(a, b, c);
	
	a = flux(tcr, { 1,2, 3, 1 }, tfloat, variable);
	a->arange(3 * 2)->printo();
	b = flux(tcr, { 2,2,3,1 }, tfloat, variable);
	b->arange(2 * 2 * 3 * 1)->printo();
	c = a->mul(b);
	//c->printo();
	mul_bw_check(a, b, c);
	
	delete tcr;
}

void cc(void)
{
	Tracer *tcr = trace(1);

	Flux *a, *b, *c;
	
	tcr->directx(1);
	a = flux(tcr, { 2,3,4,2 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2)->printo();
	b = a->transpose({ 0,1,3,2 });
	b->printo();
	trs_bw_check(a, b);
	b = a->transpose({ 3,1,0,2 });
	b->printo();
	trs_bw_check(a, b);

	a = flux(tcr, { 2,3,1,4,2 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2)->printo();
	b = a->transpose({ 3,1,0,4,2 });
	b->printo();
	trs_bw_check(a, b);
	
	/*
	a = flux(tcr, { 1,3,4,2 }, tfloat, variable);
	a->arange(1 * 3 * 4 * 2)->printo();
	b = a->transpose({ 1,0,2,3 });
	//배치 사이즈 변경후 전치 테스트
	c = flux(tcr, { 2,3,4,2 }, tfloat, variable);
	c->arange(2 * 3 * 4 * 2)->printo();
	a->copyf(c);

	b = a->transpose({ 1,0,2,3 });
	b->printo();
	trs_bw_check(a, b);
	*/
	delete tcr;
}

void dd(void)
{
	Tracer *tcr = trace(1);

	Flux *a, *c;
	vector<Flux *> *b;

	//tcr->modeset(1);
	/*
	printf("==================================================\n");
	a = flux(tcr, { 2,3,4,2 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2)->printo();
	b = a->unstack(0);
	printo(b);
	c = stack(b, 0);
	c->printo();
	printf("--------------------------------\n");
	b = a->split(2, 0);
	printo(b);
	c = concat(b, 0);
	c->printo();
	printf("==================================================\n");
	a = flux(tcr, { 2,3,4,2 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2)->printo();
	b = a->unstack(1);
	printo(b);
	c = stack(b, 1);
	c->printo();
	printf("--------------------------------\n");
	b = a->split(3, 1);
	printo(b);
	c = concat(b, 1);
	c->printo();
	printf("==================================================\n");
	a = flux(tcr, { 2,3,4,2 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2)->printo();
	b = a->unstack(2);
	printo(b);
	c = stack(b, 2);
	c->printo();
	printf("--------------------------------\n");
	b = a->split(4, 2);
	printo(b);
	c = concat(b, 2);
	c->printo();
	printf("==================================================\n");
	a = flux(tcr, { 2,3,4,2 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2)->printo();
	b = a->unstack(3);
	printo(b);
	c = stack(b, 3);
	c->printo();
	printf("--------------------------------\n");
	b = a->split(2, 3);
	printo(b);
	c = concat(b, 3);
	c->printo();
	printf("==================================================\n");
	a = flux(tcr, { 2,3,1 }, tfloat, variable);
	a->arange(2 * 3)->printo();
	b = a->unstack(0);
	printo(b);
	c = stack(b, 0);
	c->printo();
	printf("--------------------------------\n");
	b = a->split(2, 0);
	printo(b);
	c = concat(b, 0);
	c->printo();
	printf("==================================================\n");
	a = flux(tcr, { 2,3,1 }, tfloat, variable);
	a->arange(2 * 3)->printo();
	b = a->unstack(2);
	printo(b);
	c = stack(b, 2);
	c->printo();
	printf("--------------------------------\n");
	b = a->split(1, 2);
	printo(b);
	c = concat(b, 2);
	c->printo();
	
	printf("==================================================\n");
	a = flux(tcr, { 3,2,2 }, tfloat, variable);
	a->arange(3*2*2)->printo();
	vector<Flux *> l;
	l.push_back(a);
	a = flux(tcr, { 3,1,2 }, tfloat, variable);
	a->arange(3 * 1 * 2)->printo();
	l.push_back(a);
	a = flux(tcr, { 3,3,2 }, tfloat, variable);
	a->arange(3 * 3 * 2)->printo();
	l.push_back(a);
	c = concat(&l, 1);
	c->printo();
	printf("==================================================\n");
	l.clear();
	a = flux(tcr, { 1,6 }, tfloat, variable);
	a->arange(6)->printo();
	l.push_back(a);
	a = flux(tcr, { 6,6 }, tfloat, variable);
	a->arange(6*6)->printo();
	l.push_back(a);
	c = concat(&l, 0);
	c->printo();
	printf("==================================================\n");
	l.clear();
	a = flux(tcr, { 6,1 }, tfloat, variable);
	a->arange(6)->printo();
	l.push_back(a);
	a = flux(tcr, { 6,6 }, tfloat, variable);
	a->arange(6 * 6)->printo();
	l.push_back(a);
	c = concat(&l, 1);
	c->printo();
	*/
/*
	vector<Flux *> l;
	Flux *d;
	a = flux(tcr, { 2,3,2 }, tfloat, variable);
	a->arange(2 * 3 * 2)->printo();
	c = flux(tcr, { 2,1,2 }, tfloat, variable);
	c->arange(2 * 1 * 2)->printo();
	d = flux(tcr, { 2,1,2 }, tfloat, variable);
	d->arange(2 * 1 * 2)->printo();

	l.push_back(a);
	l.push_back(c);
	l.push_back(d);

	c = concat(&l, 1);
	c->shape();
	c->printo();


	a = flux(tcr, { 2,3,1 }, tfloat, variable);
	a->arange(2*3*1)->printo();
	c = flux(tcr, { 2,3,2 }, tfloat, variable);
	c->arange(2*3*2)->printo();
	

	l.push_back(a);
	l.push_back(c);

	c = concat(&l, 2);
	c->shape();
	c->printo();
	*/
	delete tcr;
}
void split_bw_check(Flux *a, vector<Flux *> *l)
{
	dumpToGrad(l);
	a->resetGrad();
	lbackward(l);
	a->printg();
	printf("\n\n");
}
void concat_bw_check(vector<Flux *> *l, Flux *b)
{
	b->dumpToGrad();
	resetGrad(l);
	b->backward();
	printg(l);
	printf("\n\n");
}
void ee(void)
{
	Tracer *tcr = trace(1);

	Flux *a, *c;
	vector<Flux *> *b;
	//tcr->modeset(1);
	
	printf("==================================================\n");
	a = flux(tcr, { 2,3,4,2 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2)->printo();
	b = a->unstack(0);
	printo(b);
	split_bw_check(a, b);
	c = stack(b, 0);
	//c->printo();
	concat_bw_check(b, c);
	printf("--------------------------------\n");
	b = a->split(2, 0);
	printo(b);
	split_bw_check(a, b);
	c = concat(b, 0);
	//c->printo();
	concat_bw_check(b, c);
	printf("==================================================\n");
	a = flux(tcr, { 2,3,4,2 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2)->printo();
	b = a->unstack(1);
	printo(b);
	split_bw_check(a, b);
	c = stack(b, 1);
	//c->printo();
	concat_bw_check(b, c);
	printf("--------------------------------\n");
	b = a->split(3, 1);
	printo(b);
	split_bw_check(a, b);
	c = concat(b, 1);
	//c->printo();
	concat_bw_check(b, c);
	printf("==================================================\n");
	a = flux(tcr, { 2,3,4,2 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2)->printo();
	b = a->unstack(2);
	printo(b);
	split_bw_check(a, b);
	c = stack(b, 2);
	//c->printo();
	concat_bw_check(b, c);
	printf("--------------------------------\n");
	b = a->split(4, 2);
	printo(b);
	split_bw_check(a, b);
	c = concat(b, 2);
	//c->printo();
	concat_bw_check(b, c);
	printf("==================================================\n");
	a = flux(tcr, { 2,3,4,2 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2)->printo();
	b = a->unstack(3);
	printo(b);
	split_bw_check(a, b);
	c = stack(b, 3);
	//c->printo();
	concat_bw_check(b, c);
	printf("--------------------------------\n");
	b = a->split(2, 3);
	printo(b);
	split_bw_check(a, b);
	c = concat(b, 3);
	//c->printo();
	concat_bw_check(b, c);
	printf("==================================================\n");
	a = flux(tcr, { 2,3,1 }, tfloat, variable);
	a->arange(2 * 3)->printo();
	b = a->unstack(0);
	printo(b);
	split_bw_check(a, b);
	c = stack(b, 0);
	//c->printo();
	concat_bw_check(b, c);
	printf("--------------------------------\n");
	b = a->split(2, 0);
	printo(b);
	split_bw_check(a, b);
	c = concat(b, 0);
	//c->printo();
	concat_bw_check(b, c);
	printf("==================================================\n");
	a = flux(tcr, { 2,3,1 }, tfloat, variable);
	a->arange(2 * 3)->printo();
	b = a->unstack(2);
	printo(b);
	split_bw_check(a, b);
	c = stack(b, 2);
	//c->printo();
	concat_bw_check(b, c);
	printf("--------------------------------\n");
	b = a->split(1, 2);
	printo(b);
	split_bw_check(a, b);
	c = concat(b, 2);
	//c->printo();
	concat_bw_check(b, c);
	
	printf("==================================================\n");
	a = flux(tcr, { 3,2,2 }, tfloat, variable);
	a->arange(3 * 2 * 2)->printo();
	vector<Flux *> l;
	l.push_back(a);
	a = flux(tcr, { 3,1,2 }, tfloat, variable);
	a->arange(3 * 1 * 2)->printo();
	l.push_back(a);
	a = flux(tcr, { 3,3,2 }, tfloat, variable);
	a->arange(3 * 3 * 2)->printo();
	l.push_back(a);
	c = concat(&l, 1);
	concat_bw_check(&l, c);
	printf("==================================================\n");
	l.clear();
	a = flux(tcr, { 1,6 }, tfloat, variable);
	a->arange(6)->printo();
	l.push_back(a);
	a = flux(tcr, { 6,6 }, tfloat, variable);
	a->arange(6*6)->printo();
	l.push_back(a);
	c = concat(&l, 0);
	concat_bw_check(&l, c);
	printf("==================================================\n");
	l.clear();
	a = flux(tcr, { 6,1 }, tfloat, variable);
	a->arange(6)->printo();
	l.push_back(a);
	a = flux(tcr, { 6,6 }, tfloat, variable);
	a->arange(6 * 6)->printo();
	l.push_back(a);
	c = concat(&l, 1);
	concat_bw_check(&l, c);

	delete tcr;
}
void aa(void) //mean square error test
{
	Tracer *tcr = trace(1);

	Flux *a, *b, *c, *w, *t;
	float f;
	/*
	a = flux(tcr, { 2,3,4,2 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2);
	c = a->softmax();
	c->printo();

	a = flux(tcr, { 2,3,4,3 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 3);
	c = a->softmax();
	c->printo();

	a = flux(tcr, { 2,3,4,4 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 4);
	c = a->softmax();
	c->printo();

	a = flux(tcr, { 2,3,4,5 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 5);
	c = a->softmax();
	c->printo();

	a = flux(tcr, { 2,3,4,6 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 6);
	c = a->softmax();
	c->printo();

	a = flux(tcr, { 2,3,4,10 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 10);
	c = a->softmax();
	c->printo();

	a = flux(tcr, { 2,3,4,11}, tfloat, variable);
	a->arange(2 * 3 * 4 * 11);
	c = a->softmax();
	c->printo();

	a = flux(tcr, { 2,3,2,2 }, tfloat, variable);
	a->arange(2 * 3 * 2*2);
	c = a->softmax();
	c->printo();

	a = flux(tcr, { 2,3,3,3 }, tfloat, variable);
	a->arange(2 * 3 * 3 * 3);
	c = a->softmax();
	c->printo();

	a = flux(tcr, { 2,3,4,4 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 4);
	c = a->softmax();
	c->printo();

	a = flux(tcr, { 2,3,5,5 }, tfloat, variable);
	a->arange(2 * 3 * 5 * 5);
	c = a->softmax();
	c->printo();

	a = flux(tcr, { 2,3,6,6 }, tfloat, variable);
	a->arange(2 * 3 * 6*6);
	c = a->softmax();
	c->printo();

	a = flux(tcr, { 2,3,3 }, tfloat, variable);
	a->arange(2 * 3 * 3);
	c = a->sum();
	c->printo();
	c = a->mean();
	c->printo();

	a = flux(tcr, { 2,4,3 }, tfloat, variable);
	a->arange(2 * 4 * 3);
	f = 1;
	b = a->plus(0, &f);
	b->printo();
	c = a->softmaxCrossEntropy(b);
	c->printo();
	c = c->mean();
	c->printo();

	a = flux(tcr, { 2,4,3 }, tfloat, variable);
	a->arange(2 * 4 * 3);
	f = 2;
	b = a->plus(0, &f);
	b->printo();
	//c = a->minus(b);
	//c->printo();
	//c = c->mul(c);
	//c->printo();
	//c = c->mean();
	//c->printo();
	c = a->meanSquareError(b);
	c->printo();
	*/
	/*
	a = flux(tcr, { 2,4,3 }, tfloat, variable);
	a->arange(-1);
	a->printo();
	w = flux(tcr, { 2,4,3 }, tfloat, trainable, "www");
	f = 0.8;
	w->fill(&f);
	w->printo();
	b = a->mul(w);
	f = 2;
	t = a->plus(0, &f);
	t->printo();
	c = b->meanSquareError(t);
	c->printo();
	a->resetGrad();
	w->resetGrad();
	c->backward();
	b->backward();
	auto op = gradient_descent_optimizer(tcr, 0.5)->minimize(c);
	op->backward();
	w->printo();
	*/

	a = flux(tcr, { 2,4,3 }, tfloat, variable);
	a->arange(-1);
	a->printo();
	w = flux(tcr, { 2,4,3 }, tfloat, trainable, nullx, "www");
	w->fill(0.8);
	w->printo();
	b = a->mul(w);
	t = a->plus(0, 2);
	t->printo();
	c = b->meanSquareError(t);

	auto op = gradient_descent_optimizer(tcr, 0.5)->minimize(c);

	tcr->run({ op });

	w->printo();

	/*
		a = flux(tcr, { 2,4,3 }, tfloat, variable);
		a->arange(-1);
		a->printo();
		w = flux(tcr, { 2,4,3 }, tfloat, variable, "www");
		f = 0.8;
		w->fill(&f);
		w->printo();
		b = a->mul(w);
		f = 2;
		t = a->plus(0, &f);
		t->printo();
		c = b->softmaxCrossEntropy(t);
		c->printo();
		a->resetGrad();
		w->resetGrad();
		c->backward();
		b->backward();
		auto op = adam_optimizer(tcr, 0.5)->minimize(c);
		op->backward();
		w->printo();
		*/
	delete tcr;
}
void bb(void) //gpu로 실힝하면 텐서프로와 비교하여 cpu보다 약간 오차가 더 발생.
{
	Tracer *ttcr = trace(1), *tcr = trace(0);

	Flux *a, *b, *c, *cc, *w, *t, *aa, *bb, *tt, *ww, *w1, *w2, *b1, *b2, *xx, *x, *y, *b3, *b4, *u;
	float f;
	/*
	aa = flux(ttcr, { 2,4,3 }, tfloat, variable);
	aa->arange(-1);
	aa->printo();
	ww = flux(ttcr, { 2,4,3 }, tfloat, variable, "www");
	f = 0.8;
	ww->fill(&f);
	ww->printo();

	bb = aa->mul(ww);
	f = 2;
	tt = aa->plus(0, &f);

	a = flux(tcr, { 2,4,3 }, tfloat, variable);
	a->feedf(aa);
	w = flux(tcr, { 2,4,3 }, tfloat, variable, "www");
	w->feedf(ww);

	t = flux(tcr, tt, variable, "bb");

	b = a->mul(w);

	c = b->softmaxCrossEntropy(t);

	auto op = adam_optimizer(tcr, 0.5)->minimize(c);

	tcr->run({ op }, { a, w });

	t->printo();
	c->printo();
	w->printo();
	*/
	/*
	aa = flux(ttcr, { 2,4,3 }, tfloat, variable);
	aa->arange(-1);
	aa->printo();
	ww = flux(ttcr, { 4,4,3 }, tfloat, variable, "www");
	f = 0.8;
	ww->fill(&f);
	ww->printo();

	a = flux(tcr, { 2,4,3 }, tfloat, variable);
	a->feedf(aa);
	w = flux(tcr, { 4,4,3 }, tfloat, variable, "www");
	w->feedf(ww);

	auto lw = w->split(2, 0);
	w1 = lw->at(0);
	w2 = lw->at(1);
	b1 = a->mul(w1);
	b2 = a->mul(w2);

	b = b1->mul(b2);

	c = b->softmaxCrossEntropy(a);

	auto op = adam_optimizer(tcr, 0.5)->minimize(c);
	tcr->run({ op });

	printf("111\n");
	c->printo();
	printf("222\n");
	w1->printo();
	printf("333\n");
	b2->printo();
	printf("444\n");
	w->printo();
	*/

	aa = flux(ttcr, { 2,4,3 }, tfloat, variable);
	aa->arange(-1);
	xx = flux(ttcr, { 4,4,3 }, tfloat, variable);
	xx->arange(-1);
	tt = flux(ttcr, { 4,4,3 }, tfloat, variable);
	tt->arange(-1);
	ww = flux(ttcr, { 4,4,3 }, tfloat, variable);
	ww->fill(0.8);

	a = flux(tcr, { 2,4,3 }, tfloat, variable, nullx, "aaa");
	a->feedf(aa);
	x = flux(tcr, { 4,4,3 }, tfloat, variable);
	x->feedf(xx);
	t = flux(tcr, { 4,4,3 }, tfloat, variable);
	t->feedf(tt);
	w = flux(tcr, { 4,4,3 }, tfloat, trainable, nullx, "www");
	w->feedf(ww);

	auto lw = w->split(2, 0);
	w1 = lw->at(0);
	w2 = lw->at(1);

	y = x->mul(w);

	b1 = a->mul(w1);
	b2 = a->mul(w2);

	b3 = concat({ b1, b2 }, 0);
	b4 = y->mul(b3);

	b = b4->tanh();
	c = b->softmaxCrossEntropy(t);

	auto op = adam_optimizer(tcr, 0.5)->minimize(c);

	//tcr->init_train();

	tcr->run({ op });

	printf("111\n");
	w->printo();
	printf("222\n");
	w1->printo();
	printf("333\n");
	w2->printo();
	printf("444\n");
	b4->printo();
	printf("555\n");
	b->printo();
	printf("666\n");
	c->printo();

	//printf("%f %f\n", sqrt(1. / (floatt)8), 1. / sqrt((floatt)8));
	/*
		aa = flux(ttcr, { 2,4,3 }, tfloat, variable);
		aa->arange(-1);
		xx = flux(ttcr, { 4,4,3 }, tfloat, variable);
		xx->arange(-1);
		tt = flux(ttcr, { 4,4,3 }, tfloat, variable);
		tt->arange(-1);
		ww = flux(ttcr, { 4,4,3 }, tfloat, variable, "www");
		ww->fill(0.8);

		a = flux(tcr, { 2,4,3 }, tfloat, variable);
		a->feedf(aa);
		x = flux(tcr, { 4,4,3 }, tfloat, variable, "456");
		x->feedf(xx);
		t = flux(tcr, { 4,4,3 }, tfloat, variable, "ttt");
		t->feedf(tt);
		w = flux(tcr, { 4,4,3 }, tfloat, trainable, "www", Initializer::xavier);

		auto lw = w->split(2, 0);
		w1 = lw->at(0);
		w2 = lw->at(1);

		y = x->mul(w);

		b1 = a->mul(w1);
		b2 = a->mul(w2);

		b3 = concat({ b1, b2 }, 0);
		b4 = y->mul(b3);

		b = b4->tanh();
		c = b->softmaxCrossEntropy(t);

		auto op = adam_optimizer(tcr, 0.5)->minimize(c);

		tcr->init_train();
		w->printo();

		tcr->run({ op });

		printf("111\n");
		w->printo();
		printf("222\n");
		w1->printo();
		printf("333\n");
		w2->printo();
		printf("444\n");
		b4->printo();
		printf("555\n");
		b->printo();
		printf("666\n");
		c->printo();
		*/


	delete tcr;
	delete ttcr;
}
void ii(void) //gpu로 실힝하면 텐서프로와 비교하여 cpu보다 약간 오차가 더 발생.
{
	Tracer *tcr = trace(1);

	Flux *a, *b, *c, *cc, *w, *t, *aa, *bb, *tt, *ww, *w1, *w2, *b1, *b2, *xx, *x, *y, *b3, *b4, *u;
	float f;

	a = flux(tcr, "[[[0, 1 ],\
					[0, 1],\
					[4, 5]],\
					[[6, 7],\
					[8, 9],\
					[10, 11]],\
					[[12, 13],\
					[14, 15],\
					[16, 17]],\
					[[18, 19],\
					[20, 21],\
					[22, 23]]]");
	a->printo();
	//a = flux(tcr, { 4,3,2 }, tint, variable);
	//a->arange(-1);
	//a->printo();
	w = flux(tcr, { 24,3 }, tfloat, trainable);
	w->arange(-1);
	w->printo();
	y = flux(tcr, { 4,3,2,3 }, tfloat, variable);
	y->fill(0.8);

	b = w->embedding_lookup(a);
	b->printo();

	x = b->mul(y);

	t = flux(tcr, { 4,3,2,3 }, tfloat, variable);
	t->fill(0.5);
	c = x->meanSquareError(t);
	//c = x->softmaxCrossEntropy(t);

	auto op = adam_optimizer(tcr, 1.0)->minimize(c);

	tcr->init_train();

	for(intt i = 0; i < 2; i++) {
		tcr->run({ op });

		printf("111\n");
		w->printo();
		printf("222\n");
		b->printo();
		printf("333\n");
		c->printo();
		printf("444\n");
		t->printo();
	}
	delete tcr;
}
void jj(void)
{
	
	Tracer *tcr = trace(1);

	Flux *a, *b, *c, *cc, *w, *t, *aa, *bb, *tt, *ww, *w1, *w2, *b1, *b2, *xx, *x, *y, *b3, *b4, *u;
	/*
	a = flux(tcr, { 2,3,4,3 }, tint, variable);
	a->arange(-1);
	a->printo();

	b = a->slice({ {0, 2}, {0,-1}, {0, 2} });
	b->printo();
	
	b = a->slice({ {0, 2}, {0,-1}, {2, 4} });
	b->printo();
	*/

	a = flux(tcr, { 4,3,2 }, tint, variable);
	a->arange(-1);
	//a->printo();

	b = a->slice({ {0,1}, {0, 2} });
	b->printo();

	b = a->slice({ {0,-1,2}, {0,-1, 2} });
	b->printo();

	b = a->slice({ {1,-2}, {1,-2} });
	b->printo();

	b = a->slice({ {1,-1}, {1,-1} });
	b->printo();
	printf("------------------------------\n");
	a->slice({ {1}, {-1} })->printo();

	a->slice({ {}, {-2}, {-2} })->printo();

	a->slice({ {1}, {1}, {1} })->printo();
	printf("------------------------------\n");
	a->slice({ {}, {-2}, {-3} })->printo();

	a->slice({ {}, {-2}, {3} })->printo();

	delete tcr;
	
	
	//Tracer *tcr;
	Flux *d, *op;

	tcr = trace(1);
	a = flux(tcr, { 4,3,2 }, tfloat, trainable);
	a->arange(-1);
	b = a->slice({ {0,1}, {0, 2} });
	w = b->feedf(trainable, Initializer::xavier);
	c = b->mul(w);
	d = c->softmaxCrossEntropy(b);
	op = adam_optimizer(tcr, 0.5)->minimize(d);

	tcr->init_train();
	for(intt i = 0; i < 1; i++) {
		tcr->run({ op });
		printf("111\n");
		w->printo();
		printf("222\n");
		b->printo();
		printf("333\n");
		a->printo();
	}
	delete tcr;

	tcr = trace(1);
	a = flux(tcr, { 4,3,2 }, tfloat, trainable);
	a->arange(-1);
	b = a->slice({ {0,-1,2}, {0,-1, 2} });
	w = b->feedf(trainable, Initializer::xavier);
	c = b->mul(w);
	d = c->softmaxCrossEntropy(b);
	op = adam_optimizer(tcr, 0.5)->minimize(d);

	tcr->init_train();
	for(intt i = 0; i < 1; i++) {
		tcr->run({ op });
		printf("111\n");
		w->printo();
		printf("222\n");
		b->printo();
		printf("333\n");
		a->printo();
	}
	delete tcr;

	tcr = trace(1);
	a = flux(tcr, { 4,3,2 }, tfloat, trainable);
	a->arange(-1);
	b = a->slice({ {1,-2}, {1,-2} });
	w = b->feedf(trainable, Initializer::xavier);
	c = b->mul(w);
	d = c->softmaxCrossEntropy(b);
	op = adam_optimizer(tcr, 0.5)->minimize(d);

	tcr->init_train();
	for(intt i = 0; i < 1; i++) {
		tcr->run({ op });
		printf("111\n");
		w->printo();
		printf("222\n");
		b->printo();
		printf("333\n");
		a->printo();
	}
	delete tcr;

	tcr = trace(1);
	a = flux(tcr, { 4,3,2 }, tfloat, trainable);
	a->arange(-1);
	b = a->slice({ {1,-1}, {1,-1} });
	w = b->feedf(trainable, Initializer::xavier);
	c = b->mul(w);
	d = c->softmaxCrossEntropy(b);
	op = adam_optimizer(tcr, 0.5)->minimize(d);

	tcr->init_train();
	for(intt i = 0; i < 1; i++) {
		tcr->run({ op });
		printf("111\n");
		w->printo();
		printf("222\n");
		b->printo();
		printf("333\n");
		a->printo();
	}
	delete tcr;

	tcr = trace(1);
	a = flux(tcr, { 4,3,2 }, tfloat, trainable);
	a->arange(-1);
	b = a->slice({ {1}, {-1} });
	w = b->feedf(trainable, Initializer::xavier);
	c = b->mul(w);
	d = c->softmaxCrossEntropy(b);
	op = adam_optimizer(tcr, 0.5)->minimize(d);

	tcr->init_train();
	for(intt i = 0; i < 1; i++) {
		tcr->run({ op });
		printf("111\n");
		w->printo();
		printf("222\n");
		b->printo();
		printf("333\n");
		a->printo();
	}
	delete tcr;

	tcr = trace(1);
	a = flux(tcr, { 4,3,2 }, tfloat, trainable);
	a->arange(-1);
	b = a->slice({ {}, {-2}, {-2} });
	w = b->feedf(trainable, Initializer::xavier);
	c = b->mul(w);
	d = c->softmaxCrossEntropy(b);
	op = adam_optimizer(tcr, 0.5)->minimize(d);

	tcr->init_train();
	for(intt i = 0; i < 1; i++) {
		tcr->run({ op });
		printf("111\n");
		w->printo();
		printf("222\n");
		b->printo();
		printf("333\n");
		a->printo();
	}
	delete tcr;

	tcr = trace(1);
	a = flux(tcr, { 4,3,2 }, tfloat, trainable);
	a->arange(-1);
	b = a->slice({ {1}, {1}, {1} });
	w = b->feedf(trainable, Initializer::xavier);
	c = b->mul(w);
	d = c->softmaxCrossEntropy(b);
	op = adam_optimizer(tcr, 0.5)->minimize(d);

	tcr->init_train();
	for(intt i = 0; i < 1; i++) {
		tcr->run({ op });
		printf("111\n");
		w->printo();
		printf("222\n");
		b->printo();
		printf("333\n");
		a->printo();
	}
	delete tcr;

	tcr = trace(1);
	a = flux(tcr, { 4,3,2 }, tfloat, trainable);
	a->arange(-1);
	b = a->slice({ {}, {-2}, {-3} });
	w = b->feedf(trainable, Initializer::xavier);
	c = b->mul(w);
	d = c->softmaxCrossEntropy(b);
	op = adam_optimizer(tcr, 0.5)->minimize(d);

	tcr->init_train();
	for(intt i = 0; i < 1; i++) {
		tcr->run({ op });
		printf("111\n");
		w->printo();
		printf("222\n");
		b->printo();
		printf("333\n");
		a->printo();
	}
	delete tcr;

	tcr = trace(1);
	a = flux(tcr, { 4,3,2 }, tfloat, trainable);
	a->arange(-1);
	b = a->slice({ {}, {-2}, {3} });
	w = b->feedf(trainable, Initializer::xavier);
	c = b->mul(w);
	d = c->softmaxCrossEntropy(b);
	op = adam_optimizer(tcr, 0.5)->minimize(d);

	tcr->init_train();
	for(intt i = 0; i < 1; i++) {
		tcr->run({ op });
		printf("111\n");
		w->printo();
		printf("222\n");
		b->printo();
		printf("333\n");
		a->printo();
	}
	
	delete tcr;
}
void mm(void)
{
	Tracer *tcr = trace(1);

	Flux *a, *b, *c;
	
	a = flux(tcr, "[[0, 2 ],\
					[3, -1]]");
	a->printo();
	printf("----------------0\n");
	b = a->one_hot(4, 5.5, 0, 0);
	b->printo();
	
	a = flux(tcr, "[[0, 2 ],\
					[3, 1]]");
	a->printo();
	printf("----------------0\n");
	b = a->one_hot(4, 5.0, 0, 0);
	b->printo();
	printf("----------------1\n");
	b = a->one_hot(3, 5.0, 0, 1);
	b->printo();
	printf("----------------2\n");
	b = a->one_hot(3, 5.0, 0, 2);
	b->printo();
	printf("----------------- -1\n");
	b = a->one_hot(3, 5.0, 0, -1);
	b->printo();
	
	printf("----------------\n");
	a = flux(tcr, "[[[0, 2 ], [3, 1]],\
					[[0, 2 ], [3, 1]]]");
	a->printo();
	printf("----------------0\n");
	b = a->one_hot(4, 5.0, 0, 0);
	b->printo();
	printf("----------------1\n");
	b = a->one_hot(3, 5.0, 0, 1);
	b->printo();
	printf("----------------2\n");
	b = a->one_hot(3, 5.0, 0, 2);
	b->printo();
	printf("----------------3\n");
	b = a->one_hot(3, 5.0, 0, 3);
	b->printo();

	a = flux(tcr, "[[[0.1, 0.3, 0.5],\
		[0.3, 0.5, 0.1]],\
		[[0.5, 0.1, 0.3],\
		[0.1, 0.3, 0.5]],\
		[[0.3, 0.5, 0.1],\
		[0.5, 0.1, 0.3]]]");
	a->printo();
	b = a->argmax(0);
	b->printo();
	b = a->argmax(1);
	b->printo();
	b = a->argmax(2);
	b->printo();
	printf("----------------\n");
	a = flux(tcr, "[ [[[0.1, 0.3, 0.5],\
		[0.3, 0.5, 0.1]],\
		[[0.5, 0.1, 0.3],\
		[0.1, 0.3, 0.5]],\
		[[0.3, 0.5, 0.1],\
		[0.5, 0.1, 0.3]]],\
		[[[0.1, 0.3, 0.5],\
		[0.3, 0.5, 0.1]],\
		[[0.5, 0.1, 0.3],\
		[0.1, 0.3, 0.5]],\
		[[0.3, 0.5, 0.1],\
		[0.5, 0.1, 0.3]]] ]");
	a->printo();
	printf("----------------0\n");
	b = a->argmax(0);
	b->printo();
	printf("----------------1\n");
	b = a->argmax(1);
	b->printo();
	printf("----------------2\n");
	b = a->argmax(2);
	b->printo();
	printf("----------------3\n");
	b = a->argmax(3);
	b->printo();

	printf("--------------------------------------\n");
	a = flux(tcr, "[[[0.1, 0.3, 0.5],\
		[0.3, 0.5, 0.1]],\
		[[0.5, 0.1, 0.3],\
		[0.1, 0.3, 0.5]],\
		[[0.3, 0.5, 0.1],\
		[0.5, 0.1, 0.3]]]");

	b = flux(tcr, "[[[0.1, 0.2, 0.5],\
		[0.3, 0.6, 0.1]],\
		[[0.5, 0.1, 0.3],\
		[0.1, 0.3, 0.5]],\
		[[0.2, 0.5, 0.1],\
		[0.5, 0.1, 0.7]]]");

	c = a->equal(b);
	c->printo();

	c = a->not_equal(b);
	c->printo();

	c = a->equal(0.5);
	c->printo();

	a = flux(tcr, { 4,3,2 }, tfloat, trainable);
	a->shape();
	a->arange(-1);
	a->printo();

	b = a->expand_dims(1);
	b->shape();
	b->printo();

	c = b->squeeze();
	c->shape();
	c->printo();

	c = b->squeeze(1);
	c->shape();
	c->printo();

	delete tcr;
}
void nn(void)
{
	Tracer *tcr;
	Flux *a, *b, *c, *t, *op, *d, *w;
	floatt f;

	tcr = trace(1);
	/*
	a = flux(tcr, { 2, 3 }, tfloat, variable);
	a->arange(2 * 3)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->dot(b, { {1}, {0} }, 0);
	c->printo();
	*/
	a = flux(tcr, { 2,4,3 }, tfloat, variable);
	a->arange(-1);
	w = flux(tcr, { 2,4,3 }, tfloat, trainable, nullx, "www");
	w->fill(0.8);
	b = a->mul(w);
	d = b->tanh();

	c = d->meanSquareError(a);

	op = gradient_descent_optimizer(tcr, 1.0)->minimize(c);

	tcr->run({ op });
	printf("------------------------------\n");
	w->printo();
	d->printo();

	delete tcr;

	tcr = trace(1);
	a = flux(tcr, { 2,4,3 }, tfloat, variable);
	a->arange(-1);
	w = flux(tcr, { 2,4,3 }, tfloat, trainable, nullx, "www");
	w->fill(0.8);
	b = a->mul(w);
	d = b->relu();

	c = d->meanSquareError(a);

	op = gradient_descent_optimizer(tcr, 1.0)->minimize(c);

	tcr->run({ op });
	printf("------------------------------\n");
	w->printo();
	d->printo();

	delete tcr;

	tcr = trace(1);
	a = flux(tcr, { 2,4,3 }, tfloat, variable);
	a->arange(-1);
	w = flux(tcr, { 2,4,3 }, tfloat, trainable, nullx, "www");
	w->fill(0.8);
	b = a->mul(w);
	d = b->sigmoid();

	c = d->meanSquareError(a);

	op = gradient_descent_optimizer(tcr, 1.0)->minimize(c);

	tcr->run({ op });
	printf("------------------------------\n");
	w->printo();
	d->printo();

	delete tcr;
}
void nn2(void)
{
	Tracer *tcr;
	Flux *a, *b, *c, *t, *op, *d, *w;
	floatt f;

	tcr = trace(1);
	//tcr->modeset(1);
	tcr->npset(10);

	a = flux(tcr, "[[[[  8  50  43]\
		[24  63  99]\
	[54  91   7]\
	[112  10  71]\
	[98  48  86]\
	[2 100  84]\
	[22  45  56]]\
\
	[[92  94  61]\
		[16  30 116]\
	[13  33  75]\
	[26 120  62]\
	[60 125  78]\
	[3  66   6]\
	[59 104  95]]\
\
	[[109  51 123]\
		[68  27  18]\
	[119  11  90]\
	[73 108  89]\
	[97   1  76]\
	[42  41   4]\
	[15  17  52]]]\
\
\
	[[[40  38   5]\
		[53 115  93]\
	[0  34  28]\
	[55  35  23]\
	[74  31 101]\
	[57  96 107]\
	[32 105  14]]\
\
		[[85  19  29]\
		[49 106  82]\
	[122 124  79]\
	[69  80  20]\
	[118  72  77]\
	[25  37  81]\
	[110  46 113]]\
\
	[[39 102  65]\
		[58  12 111]\
	[88  70  87]\
	[36 114  21]\
	[83   9 103]\
	[121  67  64]\
	[117  47  44]]]]", tfloat, trainable);
	/*
	a = flux(tcr, "[[[[29.  4.]\
		[26. 30.]\
	[32. 37.]\
	[34. 40.]]\
\
	[[7. 10.]\
		[11. 31.]\
	[33. 27.]\
	[47.  2.]]\
\
	[[46. 18.]\
		[15. 28.]\
	[22. 16.]\
	[41. 20.]]]\
\
\
	[[[42.  8.]\
		[13. 25.]\
	[5. 17.]\
	[35. 14.]]\
\
		[[38.  1.]\
		[12. 43.]\
	[24.  6.]\
	[23. 36.]]\
\
	[[21. 19.]\
		[9. 39.]\
	[45.  3.]\
	[0. 44.]]]]", tfloat, trainable);
	*/
	/*
	a = flux(tcr, "[[[[ 1.  6.]\
		[8.  9.]\
		[13.  4.]\
		[2. 14.]]]\
\
\
		[[[10.  7.]\
		[15. 11.]\
		[3.  0.]\
		[5. 12.]]]]", tfloat, trainable);
		*/
	w = flux(tcr, { 2, 3, 3 }, tfloat, trainable, Initializer::xavier, "www");
	//w->fill(0.8);
	t = flux(tcr, { 2, 3, 3 }, tfloat, trainable, Initializer::xavier, "yy");
	//t->fill(0.5);

	a->printo();
	a = a->vmax(2);
	//a = a->argmax(0);
	a->printo();

	b = a->mul(w);
	d = b->tanh();

	c = d->meanSquareError(t);

	op = gradient_descent_optimizer(tcr, 1.0)->minimize(c);

	tcr->run({ op });
	printf("------------------------------\n");
	//w->printo();
	//d->printo();

	delete tcr;
}
void sc2d(void)
{
	Tracer *tcr;
	Flux *a, *b;
	floatt f;

	tcr = trace(1);
	tcr->npset(200);
	
	a = flux(tcr, { 2,8,8,1 }, tfloat, variable);
	a->arange(-1);
	a->printo(1, 1);

	b = a->scoopup(4, 4, 3, 3);
	b->printo(1, 1);
	printf("============================================\n");
	b->scoopup(3, 2);
	b->printo(1, 1);
	printf("============================================\n");
	b->scoopup(1, 1);
	b->printo(1, 1);
	printf("============================================\n");
	b->scoopup(1, 3);
	b->printo(1, 1);
	printf("============================================\n");
	a = flux(tcr, { 2,5,8,1 }, tfloat, variable);
	a->arange(-1);
	a->printo(1, 1);

	b = a->scoopup(4, 3, 1, 3);
	b->printo(1, 1);
	printf("============================================\n");
	a = flux(tcr, { 2,8,1 }, tfloat, variable);
	a->arange(-1);
	a->printo(1, 1);

	b = a->scoopup(0, 4, 0, 3);
	b->printo(1, 1);
	printf("\n");
}
void sc2d2(void)
{
	Tracer *tcr;
	Flux *a, *b;
	floatt f;

	tcr = trace(1);
	tcr->npset(200);

	a = flux(tcr, { 2,8,8,3 }, tfloat, variable);
	a->arange(-1);
	a->printo(1, 1);

	b = a->scoopup(4, 4, 3, 3);
	b->printo(1, 1);
	printf("============================================\n");
	b->scoopup(3, 2);
	b->printo(1, 1);
	printf("============================================\n");
	b->scoopup(1, 1);
	b->printo(1, 1);
	printf("============================================\n");
	b->scoopup(1, 3);
	b->printo(1, 1);



	printf("============================================\n");
	a = flux(tcr, { 2,5,8,3 }, tfloat, variable);
	a->arange(-1);
	a->printo(1, 1);

	b = a->scoopup(4, 3, 1, 3);
	b->printo(1, 1);
	printf("============================================\n");
	a = flux(tcr, { 2,8,1 }, tfloat, variable);
	a->arange(-1);
	a->printo(1, 1);

	b = a->scoopup(0, 4, 0, 3);
	b->printo(1, 1);
	printf("\n");
}

void sc2d3(void)
{
	Tracer *tcr;
	Flux *a, *b;
	floatt f;

	tcr = trace(1);
	tcr->npset(200);


	a = flux(tcr, { 5,8,8,2 }, tfloat, variable);
	a->arange(-1);
	a->printo(1, 1);

	b = a->scoopup(4, 4, 3, 3);
	b->printo(1, 1);
	printf("============================================\n");
	b->scoopup(3, 2);
	b->printo(1, 1);
	printf("============================================\n");
	b->scoopup(1, 1);
	b->printo(1, 1);
	printf("============================================\n");
	b->scoopup(1, 3);
	b->printo(1, 1);



	printf("============================================\n");
	a = flux(tcr, { 5,5,8,2 }, tfloat, variable);
	a->arange(-1);
	a->printo(1, 1);

	b = a->scoopup(4, 3, 1, 3);
	b->printo(1, 1);
	printf("============================================\n");
	a = flux(tcr, { 2,8,1 }, tfloat, variable);
	a->arange(-1);
	a->printo(1, 1);

	b = a->scoopup(0, 4, 0, 3);
	b->printo(1, 1);
	printf("\n");
}
void sc1d(void)
{
	Tracer *tcr;
	Flux *a, *b;
	floatt f;

	tcr = trace(1);
	tcr->npset(200);

	a = flux(tcr, { 2,64,1 }, tfloat, variable);
	a->arange(-1);
	a->printo(1, 1);

	b = a->scoopup(-1, 4, -1, 3);
	b->printo(1, 1);
	printf("============================================\n");
	b->scoopup(3, 2);
	b->printo(1, 1);
	printf("============================================\n");
	b->scoopup(1, 1);
	b->printo(1, 1);
	printf("============================================\n");
	b->scoopup(1, 3);
	b->printo(1, 1);
	printf("============================================\n");
	a = flux(tcr, { 2,40,1 }, tfloat, variable);
	a->arange(-1);
	a->printo(1, 1);

	b = a->scoopup(-1, 3, -1, 3);
	b->printo(1, 1);
	printf("============================================\n");
	a = flux(tcr, { 2,8,1 }, tfloat, variable);
	a->arange(-1);
	a->printo(1, 1);

	b = a->scoopup(0, 4, 0, 3);
	b->printo(1, 1);
	printf("\n");
}
void sc1d2(void)
{
	Tracer *tcr;
	Flux *a, *b;
	floatt f;

	tcr = trace(1);
	tcr->npset(200);

	a = flux(tcr, { 2,64,3 }, tfloat, variable);
	a->arange(-1);
	a->printo(1, 1);

	b = a->scoopup(-1, 4, -1, 3);
	b->printo(1, 1);
	printf("============================================\n");
	b->scoopup(3, 2);
	b->printo(1, 1);
	printf("============================================\n");
	b->scoopup(1, 1);
	b->printo(1, 1);
	printf("============================================\n");
	b->scoopup(1, 3);
	b->printo(1, 1);



	printf("============================================\n");
	a = flux(tcr, { 2,40,3 }, tfloat, variable);
	a->arange(-1);
	a->printo(1, 1);

	b = a->scoopup(-1, 3, -1, 3);
	b->printo(1, 1);

	printf("\n");
}

void sc1d3(void)
{
	Tracer *tcr;
	Flux *a, *b;
	floatt f;

	tcr = trace(1);
	tcr->npset(200);


	a = flux(tcr, { 5,64,2 }, tfloat, variable);
	a->arange(-1);
	a->printo(1, 1);

	b = a->scoopup(-1, 4, -1, 3);
	b->printo(1, 1);
	printf("============================================\n");
	b->scoopup(3, 2);
	b->printo(1, 1);
	printf("============================================\n");
	b->scoopup(1, 1);
	b->printo(1, 1);
	printf("============================================\n");
	b->scoopup(1, 3);
	b->printo(1, 1);



	printf("============================================\n");
	a = flux(tcr, { 5,40,2 }, tfloat, variable);
	a->arange(-1);
	a->printo(1, 1);

	b = a->scoopup(-1, 3, -1, 3);
	b->printo(1, 1);

	printf("\n");
}
void ww(void)
{
	Tracer *tcr;
	Flux *a, *b;
	floatt f;

	tcr = trace(1);
	//tcr->modeset(1);

	a = flux(tcr, { 4,3,4,2 }, tfloat, variable);
	a->arange(-1);
	a->printo(1, 1);

	a = a->mean(1);
	a->printo(1, 1);
}
void hh(void)
{
	Tracer *tcr, *tcr2;

	Flux *a, *b, *c, *a2, *b2, *c2, *op;
	Flux *w, *t, *d;

	/*
	tcr = trace(1);
	tcr2 = trace(-1);
	a = flux(tcr, { 2, 3 }, tfloat, variable);
	a->arange(2 * 3)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->dot(b, { {1}, {0} }, 0);
	c->printo();

	tcr2->promptMode(1);
	tcr->portingGraph(tcr2);
	c2 = tcr2->getFlux(c);
	c2->printo();

	delete tcr;
	delete tcr2;

	tcr = trace(1);
	tcr2 = trace(-1);

	a = flux(tcr, { 3, 2 }, tfloat, variable);
	a->arange(3 * 2)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->mul(b);
	c->printo();

	tcr2->promptMode(1);
	tcr->portingGraph(tcr2);
	c2 = tcr2->getFlux(c);
	c2->printo();

	delete tcr;
	delete tcr2;

	tcr = trace(1);
	tcr2 = trace(-1);
	vector<Flux *> *l;

	a = flux(tcr, { 2,3,4,2 }, tfloat, variable);
	a->arange(2 * 3 * 4 * 2)->printo();
	l = a->unstack(0);
	printo(l);
	b = stack(l, 0);
	b->printo();
	l = a->split(2, 0);
	printo(l);
	c = concat(l, 0);
	c->printo();

	tcr2->promptMode(1);
	tcr->portingGraph(tcr2);
	b2 = tcr2->getFlux(b);
	b2->printo();
	c2 = tcr2->getFlux(c);
	c2->printo();

	delete tcr;
	delete tcr2;

	tcr = trace(1);
	tcr2 = trace(-1);

	a = flux(tcr, { 2,4,3 }, tfloat, variable);
	a->arange(-1);
	a->printo();
	w = flux(tcr, { 2,4,3 }, tfloat, trainable, nullx, "www");
	w->fill(0.8);
	w->printo();
	b = a->mul(w);
	t = a->plus(0, 2);
	t->printo();
	c = b->meanSquareError(t);

	op = gradient_descent_optimizer(tcr, 0.5)->minimize(c);

	tcr->run({ op });

	w->printo();

	tcr->portingGraph(tcr2);
	b2 = tcr2->getFlux(op);
	tcr2->run({ b2 });
	b2 = tcr2->getFlux(w);
	b2->printo();

	delete tcr;
	delete tcr2;

	Tracer *ttcr = trace(1);
	tcr = trace(0);

	Flux *cc, *aa, *bb, *tt, *ww, *w1, *w2, *b1, *xx, *x, *y, *b3, *b4, *u;
	float f;
	aa = flux(ttcr, { 2,4,3 }, tfloat, variable);
	aa->arange(-1);
	xx = flux(ttcr, { 4,4,3 }, tfloat, variable);
	xx->arange(-1);
	tt = flux(ttcr, { 4,4,3 }, tfloat, variable);
	tt->arange(-1);
	ww = flux(ttcr, { 4,4,3 }, tfloat, variable);
	ww->fill(0.8);

	a = flux(tcr, { 2,4,3 }, tfloat, variable, nullx, "aaa");
	a->feedf(aa);
	x = flux(tcr, { 4,4,3 }, tfloat, variable);
	x->feedf(xx);
	t = flux(tcr, { 4,4,3 }, tfloat, variable);
	t->feedf(tt);
	w = flux(tcr, { 4,4,3 }, tfloat, trainable, nullx, "www");
	w->feedf(ww);

	auto lw = w->split(2, 0);
	w1 = lw->at(0);
	w2 = lw->at(1);

	y = x->mul(w);

	b1 = a->mul(w1);
	b2 = a->mul(w2);

	b3 = concat({ b1, b2 }, 0);
	b4 = y->mul(b3);

	b = b4->tanh();
	c = b->softmaxCrossEntropy(t);

	op = adam_optimizer(tcr, 0.5)->minimize(c);

	//tcr->init_train();

	tcr->run({ op });
	w->printo();

	tcr2 = trace(-1);

	tcr->portingGraph(tcr2);
	a = tcr2->getFlux(a);
	x = tcr2->getFlux(x);
	t = tcr2->getFlux(t);
	w = tcr2->getFlux(w);
	a->feedf(aa);
	x->feedf(xx);
	t->feedf(tt);
	w->feedf(ww);

	op = tcr2->getFlux(op);
	tcr2->run({ op });
	w->printo();

	delete tcr;
	delete tcr2;

	tcr = trace(1);
	tcr2 = trace(-1);

	a = flux(tcr, "[[[0, 1 ],\
					[0, 1],\
					[4, 5]],\
					[[6, 7],\
					[8, 9],\
					[10, 11]],\
					[[12, 13],\
					[14, 15],\
					[16, 17]],\
					[[18, 19],\
					[20, 21],\
					[22, 23]]]");
	a->printo();
	w = flux(tcr, { 24,3 }, tfloat, trainable);
	w->arange(-1);
	w->printo();
	y = flux(tcr, { 4,3,2,3 }, tfloat, variable);
	y->fill(0.8);

	b = w->embedding_lookup(a);
	b->printo();

	x = b->mul(y);

	t = flux(tcr, { 4,3,2,3 }, tfloat, variable);
	t->fill(0.5);
	c = x->meanSquareError(t);
	//c = x->softmaxCrossEntropy(t);

	op = adam_optimizer(tcr, 1.0)->minimize(c);

	tcr->init_train();

	tcr->run({ op });
	w->printo();

	tcr->portingGraph(tcr2);
	aa = tcr2->getFlux(a);
	aa->feedf(a);
	op = tcr2->getFlux(op);
	tcr2->run({ op });
	w = tcr2->getFlux(w);
	w->printo();

	delete tcr;
	delete tcr2;

	Flux *d;

	tcr = trace(1);
	tcr2 = trace(-1);

	a = flux(tcr, { 4,3,2 }, tfloat, trainable);
	a->arange(-1);
	b = a->slice({ {0,1}, {0, 2} });
	w = b->feedf(trainable, Initializer::he);
	c = b->mul(w);
	d = c->softmaxCrossEntropy(b);
	op = adam_optimizer(tcr, 0.5)->minimize(d);

	tcr->init_train();
	tcr->run({ op });
	w->printo();

	tcr->portingGraph(tcr2);
	tcr2->init_train();
	op = tcr2->getFlux(op);
	tcr2->run({ op });
	w = tcr2->getFlux(w);
	w->printo();

	delete tcr;
	delete tcr2;
	*/

	tcr = trace(1);
	tcr2 = trace(-1);
	a = flux(tcr, "[[0, 2 ],\
						[3, -1]]");
	b = a->one_hot(4, 5.5, 0, 0);
	b->printo();

	tcr->portingGraph(tcr2);
	a2 = tcr2->getFlux(a);
	a2->feedf(a);
	b = tcr2->getFlux(b);
	tcr2->run({ b });
	b->printo();
	delete tcr;
	delete tcr2;

	tcr = trace(1);
	tcr2 = trace(-1);
	a = flux(tcr, "[[[0.1, 0.3, 0.5],\
		[0.3, 0.5, 0.1]],\
		[[0.5, 0.1, 0.3],\
		[0.1, 0.3, 0.5]],\
		[[0.3, 0.5, 0.1],\
		[0.5, 0.1, 0.3]]]");
	b = a->argmax(0);
	b->printo();

	tcr->portingGraph(tcr2);
	a2 = tcr2->getFlux(a);
	a2->feedf(a);
	b = tcr2->getFlux(b);
	tcr2->run(b);
	b->printo();
	delete tcr;
	delete tcr2;

	tcr = trace(1);
	tcr2 = trace(-1);
	a = flux(tcr, "[[[0.1, 0.3, 0.5],\
		[0.3, 0.5, 0.1]],\
		[[0.5, 0.1, 0.3],\
		[0.1, 0.3, 0.5]],\
		[[0.3, 0.5, 0.1],\
		[0.5, 0.1, 0.3]]]");

	b = flux(tcr, "[[[0.1, 0.2, 0.5],\
		[0.3, 0.6, 0.1]],\
		[[0.5, 0.1, 0.3],\
		[0.1, 0.3, 0.5]],\
		[[0.2, 0.5, 0.1],\
		[0.5, 0.1, 0.7]]]");

	c = a->equal(b);
	c->printo();

	d = a->not_equal(b);
	d->printo();

	tcr->portingGraph(tcr2);
	a2 = tcr2->getFlux(a);
	a2->feedf(a);
	b2 = tcr2->getFlux(b);
	b2->feedf(b);
	c = tcr2->getFlux(c);
	d = tcr2->getFlux(d);
	tcr2->run({ c, d });
	c->printo();
	d->printo();
	delete tcr;
	delete tcr2;

	floatt f;

	tcr = trace(1);
	tcr2 = trace(-1);
	a = flux(tcr, { 2,4,3 }, tfloat, variable);
	a->arange(-1);
	w = flux(tcr, { 2,4,3 }, tfloat, trainable, nullx, "www");
	w->fill(0.8);
	b = a->mul(w);
	d = b->tanh();

	c = d->meanSquareError(a);

	op = gradient_descent_optimizer(tcr, 1.0)->minimize(c);

	tcr->run({ op });
	w->printo();
	d->printo();

	tcr->portingGraph(tcr2);
	tcr2->init_train();
	op = tcr2->getFlux(op);
	tcr2->run({ op });
	w = tcr2->getFlux(w);
	w->printo();
	d = tcr2->getFlux(d);
	d->printo();

	delete tcr;
	delete tcr2;

	tcr = trace(1);
	tcr2 = trace(-1);
	a = flux(tcr, { 2,4,3 }, tfloat, variable);
	a->arange(-1);
	w = flux(tcr, { 2,4,3 }, tfloat, trainable, nullx, "www");
	w->fill(0.8);
	b = a->mul(w);
	d = b->relu();

	c = d->meanSquareError(a);

	op = gradient_descent_optimizer(tcr, 1.0)->minimize(c);

	tcr->run({ op });
	w->printo();
	d->printo();

	tcr->portingGraph(tcr2);
	tcr2->init_train();
	op = tcr2->getFlux(op);
	tcr2->run({ op });
	w = tcr2->getFlux(w);
	w->printo();
	d = tcr2->getFlux(d);
	d->printo();

	delete tcr;
	delete tcr2;

	tcr = trace(1);
	tcr2 = trace(-1);
	a = flux(tcr, { 2,4,3 }, tfloat, variable);
	a->arange(-1);
	w = flux(tcr, { 2,4,3 }, tfloat, trainable, nullx, "www");
	w->fill(0.8);
	b = a->mul(w);
	d = b->sigmoid();

	c = d->meanSquareError(a);

	op = gradient_descent_optimizer(tcr, 1.0)->minimize(c);

	tcr->run({ op });
	w->printo();
	d->printo();

	tcr->portingGraph(tcr2);
	tcr2->init_train();
	op = tcr2->getFlux(op);
	tcr2->run({ op });
	w = tcr2->getFlux(w);
	w->printo();
	d = tcr2->getFlux(d);
	d->printo();

	delete tcr;
	delete tcr2;
}
void performance_test(intt sz)
{
	Tracer *tcr = trace(1);
	/*
	Flux *a = flux(tcr, "[[[0, 1 ],\
					[0, -1],\
					[-2, -3]],\
					[[6, 7],\
					[8, 9],\
					[-10, 11]],\
					[[12, 13],\
					[14, -15],\
					[16, 17]],\
					[[18, 19],\
					[-20, 21],\
					[22, 23]]]");
	a = a->clipValue(-3, 18);
	a->printo();
	*/
	/*Flux *a = flux(tcr, { 4, 4 }, tfloat, variable);
	a->arange(4 * 4)->printo();
	Flux *b = flux(tcr, { 4, 4 }, tfloat, variable);
	b->arange(4 * 4)->printo();
	Flux *c = a->dot(b, { {1}, {0} }, 0);
	c->printo();*/
	
	//tcr->modeset(1);
	//tcr->lapset(1);
	printf("11111111111\n");
	tcr->directx(1);
	intt x = 10000;
	Flux *a = flux(tcr, { x, x }, tfloat, variable), *c;
	a->arange(x * x)->printo();
	Flux *b = flux(tcr, { x, x }, tfloat, variable);
	b->arange(x * x)->printo();
	c = a->dot(b, { {1}, {0} }, 0);
	for(intt i = 0;i < 100; i++) {
		unit lap = xucurrenttime();
		c->exec();
		printf("step: %d lap: %f\n", i, (xucurrenttime() - lap) / 1000000.0);
		c->printo();
	}
	/*
	if(sz == 0) sz = 500;
	Flux *a = flux(tcr, { sz, sz }, tfloat, variable);
	a->randu(0, 0.5)->printo();
	Flux *b = flux(tcr, { sz, sz }, tfloat, variable);
	b->randu(0, 0.5)->printo();	
	unit lap = xucurrenttime();
	Flux *c = a->dot(b, { {1}, {0} }, 0);
	printf("run#5 lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
	c->printo();
	*/
	/*
	Flux *a = flux(tcr, { 2000, 30, 20 }, tfloat, variable);
	a->arange(2000 * 30 * 20)->printo();
	Flux *b = flux(tcr, { 30,20 }, tfloat, variable);
	b->arange(30 * 20)->printo();
	unit lap = xucurrenttime();
	Flux *c = a->mul(b);
	printf("run#5 lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
	c->printo();
	c->shape();
	*/
	delete tcr;
}
void qq(void)
{
	Tracer *tcr = trace(1);

	Flux *a, *b, *c;
	/*
	a = flux(tcr, { 2, 3 }, tfloat, variable);
	a->arange(2 * 3)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->dot(b, { {1}, {0} }, 0);
	c->printo();

	a = flux(tcr, { 2, 3 }, tfloat, variable);
	a->arange(2 * 3)->printo();
	a = a->transpose({ 1, 0 });
	a->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->dot(b, { {0}, {0} }, 1);
	c->printo();

	a = flux(tcr, { 2, 3 }, tfloat, variable);
	a->arange(2 * 3)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	b = b->transpose({ 1, 0 });
	b->printo();
	c = a->dot(b, { {1}, {1} }, 2);
	c->printo();
	*/

	unit lap;
	/*
	a = flux(tcr, { 2000, 4000 }, tfloat, variable);
	a->arange(2000 * 4000);
	b = flux(tcr, { 4000,3000 }, tfloat, variable);
	b->arange(4000 * 3000);
	lap = xucurrenttime();
	c = a->dot(b, { {1}, {0} }, 0);
	printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
	c->printo();

	a = flux(tcr, { 2000, 4000 }, tfloat, variable);
	a->arange(2000 * 4000);
	a = a->transpose({ 1, 0 });
	a->printo();
	b = flux(tcr, { 4000,3000 }, tfloat, variable);
	b->arange(4000 * 3000);
	lap = xucurrenttime();
	c = a->dot(b, { {0}, {0} }, 1);
	printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
	c->printo();


	a = flux(tcr, { 2000, 4000 }, tfloat, variable);
	a->arange(2000 * 4000);
	b = flux(tcr, { 4000,3000 }, tfloat, variable);
	b->arange(4000 * 3000);
	lap = xucurrenttime();
	b = b->transpose({ 1, 0 });
	printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
	lap = xucurrenttime();
	c = a->dot(b, { {1}, {1} }, 2);
	printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
	c->printo();
	*/
	/*
	a = flux(tcr, { 10000, 10000 }, tfloat, variable);
	a->randu(0, 0.5);
	b = flux(tcr, { 10000,10000 }, tfloat, variable);
	b->randu(0, 0.5);
	lap = xucurrenttime();
	c = a->dot(b, { {1}, {0} }, 0);
	printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
	c->printo();

	a = flux(tcr, { 10000, 10000 }, tfloat, variable);
	lap = xucurrenttime();
	a->randu(0, 0.5);
	printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
	a->printo();
	lap = xucurrenttime();
	a = a->transpose({ 1, 0 });
	printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
	a->printo();
	b = flux(tcr, { 10000,10000 }, tfloat, variable);
	b->randu(0, 0.5);
	lap = xucurrenttime();
	c = a->dot(b, { {0}, {0} }, 1);
	printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
	c->printo();


	a = flux(tcr, { 10000, 10000 }, tfloat, variable);
	a->randu(0, 0.5);
	b = flux(tcr, { 10000,10000 }, tfloat, variable);
	b->randu(0, 0.5);
	lap = xucurrenttime();
	b = b->transpose({ 1, 0 });
	printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
	lap = xucurrenttime();
	c = a->dot(b, { {1}, {1} }, 2);
	printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
	c->printo();
	*/

	Flux *d, *e;
	/*
	a = flux(tcr, { 2, 3, 4 }, tfloat, variable);
	a->arange(2 * 3 * 4)->printo();
	b = flux(tcr, { 4,5 }, tfloat, variable);
	b->arange(4 *5)->printo();
	c = a->dot(b, { {2}, {0} }, 0);
	c->printo();

	d = c->dot(b, { {2}, {1} });
	d->printo();
	e = a->dot(c, { {0, 1}, {0, 1} });
	e->printo();


	a = flux(tcr, { 2, 3, 5 }, tfloat, variable);
	a->arange(2 * 3 * 5)->printo();
	b = flux(tcr, { 2,3,4 }, tfloat, variable);
	b->arange(2 * 3 *4)->printo();
	c = a->dot(b, { {0,1}, {0, 1} });
	c->printo();

	d = b->dot(c, { {2}, {1} });
	d->printo();
	e = a->dot(c, { {2}, {0} });
	e->printo();

	a = flux(tcr, { 2, 3, 5 }, tfloat, variable);
	a->arange(2 * 3 * 5)->printo();
	b = flux(tcr, { 4,2,5 }, tfloat, variable);
	b->arange(4*2*5)->printo();
	c = a->dot(b, { {2}, {2} });
	c->printo();

	d = c->dot(b, { {2,3}, {0,1} });
	d->printo();
	e = c->dot(a, { {0,1}, {0,1} });
	e->printo();

	a = flux(tcr, { 2, 3, 5 }, tfloat, variable);
	a->arange(2 * 3 * 5)->printo();
	b = flux(tcr, { 4,2,3 }, tfloat, variable);
	b->arange(4*2*3)->printo();
	c = a->dot(b, { {0,1}, {1, 2} });
	c->printo();

	d = b->dot(c, { {0}, {1} });
	d->printo();
	e = c->dot(a, { {0}, {2} });
	e->printo();

	a = flux(tcr, { 2, 3, 5 }, tfloat, variable);
	a->arange(2 * 3 * 5)->printo();
	b = flux(tcr, { 4,2,3 }, tfloat, variable);
	b->arange(4 * 2 * 3)->printo();
	c = a->dot(b, { {0,1}, {2,1} });
	c->printo();

	d = b->dot(c, { {0}, {1} });
	d->printo();
	e = c->dot(a, { {0}, {2} });
	e->printo();
	*/

	/*
	a = flux(tcr, { 2, 3, 4 }, tfloat, variable);
	a->arange(2 * 3 * 4)->printo();
	b = flux(tcr, { 4,5 }, tfloat, variable);
	b->arange(4 * 5)->printo();
	c = a->dot(b, { {2}, {0} }, 0);
	c->printo();

	dot_bw_check(a, b, c, 0);

	a = flux(tcr, { 2, 3, 5 }, tfloat, variable);
	a->arange(2 * 3 * 5)->printo();
	b = flux(tcr, { 2,3,4 }, tfloat, variable);
	b->arange(2 * 3 * 4)->printo();
	c = a->dot(b, { {0,1}, {0, 1} });
	c->printo();

	dot_bw_check(a, b, c, 0);

	a = flux(tcr, { 2, 3, 5 }, tfloat, variable);
	a->arange(2 * 3 * 5)->printo();
	b = flux(tcr, { 4,2,5 }, tfloat, variable);
	b->arange(4 * 2 * 5)->printo();
	c = a->dot(b, { {2}, {2} });
	c->printo();

	dot_bw_check(a, b, c, 0);

	a = flux(tcr, { 2, 3, 5 }, tfloat, variable);
	a->arange(2 * 3 * 5)->printo();
	b = flux(tcr, { 4,2,3 }, tfloat, variable);
	b->arange(4 * 2 * 3)->printo();
	c = a->dot(b, { {0,1}, {1, 2} });
	c->printo();

	dot_bw_check(a, b, c, 0);

	a = flux(tcr, { 2, 3, 5 }, tfloat, variable);
	a->arange(2 * 3 * 5)->printo();
	b = flux(tcr, { 4,2,3 }, tfloat, variable);
	b->arange(4 * 2 * 3)->printo();
	c = a->dot(b, { {0,1}, {2,1} });
	c->printo();

	dot_bw_check(a, b, c, 0);
	*/
	/*
		intt x = 1000;
		a = flux(tcr, { x, x }, tfloat, variable);
		a->arange(x * x);
		b = flux(tcr, { x,x }, tfloat, variable);
		b->arange(x * x);

		lap = xucurrenttime();
		c = a->dot(b, { {1}, {0} });
		printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
		c->printo();
		*/
		/*
			intt x = 10000, i;
			a = flux(tcr, { x, x }, tfloat, variable);
			a->arange(x * x);

			for(i = 0;i < 2; i++) {
				lap = xucurrenttime();
				b = a->transpose({ 1, 0 });
				printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
			}
			b->printo();
			*/
			/*
				intt x = 10000, i;
				a = flux(tcr, { x, x }, tint, variable);
				a->arange(x * x);
				b = a->transpose({ 1, 0 });
				for(i = 0;i < 2; i++) {
					lap = xucurrenttime();
					tcr->run({ a });
					printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
				}
				b->shape();
				b->printo();
				*/
				/*
					intt x = 10000, i;
					a = flux(tcr, { x, x }, tint, variable);
					a->arange(x * x);

					b = flux(tcr, { x }, tint, variable);
					b->arange(x);
					c = a->mul(b);
					for(i = 0;i < 2; i++) {
						lap = xucurrenttime();
						tcr->run({ c });
						printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
					}
					c->shape();
					c->printo();
					*/

	intt x = 1000, i;
	a = flux(tcr, { x, x }, tfloat, variable);
	a->arange(x * x);

	b = flux(tcr, { x, x }, tfloat, variable);
	b->arange(x * x);
	c = a->dot(b, { {1}, {0} });
	for(i = 0;i < 2; i++) {
		lap = xucurrenttime();
		tcr->run({ c });
		printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
	}
	c->shape();
	c->printo();

	delete tcr;
}
void rr(void)
{
	Tracer *tcr = trace(1);

	Flux *a, *b, *c, *w, *t, *op;
	
	tcr->modeset(1);
	/*
	a = flux(tcr, { 3,2 }, tfloat, variable);
	a->arange(-1);
	a->printo();
	b = flux(tcr, { 2,3 }, tfloat, variable);
	b->arange(-1);
	b->printo();
	c = a->matmul(b);
	c->printo();
	
	a = flux(tcr, {4,3,2 }, tfloat, variable);
	a->arange(-1);
	b = flux(tcr, { 4,2,3 }, tfloat, variable);
	b->arange(-1);
	c = a->matmul(b);
	c->printo();
	

	a = flux(tcr, { 4,2,3 }, tfloat, variable);
	a->arange(-1);
	b = flux(tcr, { 4,2,3 }, tfloat, variable);
	b->arange(-1);
	c = a->matmul(b, 1);
	c->printo();
	
	a = flux(tcr, { 4,3,2 }, tfloat, variable);
	a->arange(-1);
	b = flux(tcr, { 4,3,2 }, tfloat, variable);
	b->arange(-1);
	c = a->matmul(b, 2);
	c->printo();
	
	a = flux(tcr, { 4,2,3 }, tfloat, variable);
	a->arange(-1);
	b = flux(tcr, { 4,3,2 }, tfloat, variable);
	b->arange(-1);
	c = a->matmul(b, 3);
	c->printo();
	*/
	/*
	a = flux(tcr, { 4,5,3 }, tfloat, variable);
	a->arange(-1);
	b = flux(tcr, { 4,3,5 }, tfloat, variable);
	b->arange(-1);
	c = a->matmul(b);
	c->printo();
	
	a = flux(tcr, { 4,5,3 }, tfloat, variable);
	a->arange(-1);
	b = flux(tcr, { 4,5,3 }, tfloat, variable);
	b->arange(-1);
	c = a->matmul(b, 1);
	c->printo();
	
	a = flux(tcr, { 4,3,5 }, tfloat, variable);
	a->arange(-1);
	b = flux(tcr, { 4,3,5 }, tfloat, variable);
	b->arange(-1);
	c = a->matmul(b, 2);
	c->printo();

	a = flux(tcr, { 4,3,5 }, tfloat, variable);
	a->arange(-1);
	b = flux(tcr, { 4,5,3 }, tfloat, variable);
	b->arange(-1);
	c = a->matmul(b, 3);
	c->printo();
	*/
	/*
	a = flux(tcr, { 16,16,7 }, tfloat, variable);
	a->arange(-1);
	b = flux(tcr, { 16,7,16 }, tfloat, trainable, Initializer::xavier);
	b->arange(-1);
	c = a->matmul(b);
	c->printo();

	a = flux(tcr, { 16,7,16 }, tfloat, variable);
	a->arange(-1);
	b = flux(tcr, { 16,7,16 }, tfloat, trainable, Initializer::xavier);
	b->arange(-1);
	c = a->matmul(b, 1);
	c->printo();
	
	a = flux(tcr, { 16,16,7 }, tfloat, variable);
	a->arange(-1);
	b = flux(tcr, { 16,16,7 }, tfloat, trainable, Initializer::xavier);
	b->arange(-1);
	c = a->matmul(b, 2);
	c->printo();
	*/
	a = flux(tcr, { 16,7,16 }, tfloat, variable);
	a->arange(-1);
	b = flux(tcr, { 16,16,7 }, tfloat, trainable, Initializer::xavier);
	b->arange(-1);
	c = a->matmul(b, 3);
	c->printo();
	/*
	//성능데스트
	a = flux(tcr, { 4,30,200 }, tfloat, variable);
	a->arange(-1);
	b = flux(tcr, { 4,200,30 }, tfloat, variable);
	b->arange(-1);
	unit lap = xucurrenttime();
	c = a->matmul(b);
	printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
	c->printo();
	*/
	
	//이하 역전파 기울기 데스트
	/*
	t = flux(tcr, { 4,5,10 }, tfloat, variable);
	t->randn(0, 1);
	a = flux(tcr, { 4,5,3 }, tfloat, variable);
	a->randn(0, 1);
	b = flux(tcr, { 3,10 }, tfloat, trainable, Initializer::xavier);
	c = a->dot(b, { {2}, {0} });
	auto bb = flux(tcr, { 10,10 }, tfloat, trainable, Initializer::xavier);
	c = c->dot(bb, { {2}, {0} });
	*/
	/*
	t = flux(tcr, { 4,5,5 }, tfloat, variable);
	t->randn(0, 1);
	a = flux(tcr, { 4,5,3 }, tfloat, variable);
	a->randn(0, 1);
	b = flux(tcr, { 4,3,5 }, tfloat, trainable, Initializer::xavier);
	c = a->matmul(b);
	//auto bb = flux(tcr, { 4,5,5 }, tfloat, trainable, Initializer::xavier);
	//c = c->matmul(bb);
	
	/*
	t = flux(tcr, { 3,3 }, tfloat, variable);
	t->randn(0, 1);
	a = flux(tcr, { 3,2 }, tfloat, variable);
	a->randn(0, 1);
	b = flux(tcr, { 2,3 }, tfloat, trainable, Initializer::xavier);
	c = a->matmul(b);
	*/
/*
	t = flux(tcr, { 8,4 }, tfloat, variable);
	t->randn(0, 1);
	a = flux(tcr, { 8,3 }, tfloat, variable);
	a->randn(0, 1);
	b = flux(tcr, { 3,4 }, tfloat, trainable, Initializer::xavier);
	c = a->matmul(b);
	//c = a->dot(b, { {1}, {0} });
	*/
/*
	t = flux(tcr, { 80,40 }, tfloat, variable);
	t->randn(0, 1);
	a = flux(tcr, { 30,80 }, tfloat, variable);
	a->randn(0, 1);
	b = flux(tcr, { 30,40 }, tfloat, trainable, Initializer::xavier);
	c = a->matmul(b, 1);
	*/
/*
	t = flux(tcr, { 80,40 }, tfloat, variable);
	t->randn(0, 1);
	a = flux(tcr, { 80,30 }, tfloat, variable);
	a->randn(0, 1);
	b = flux(tcr, { 40,30 }, tfloat, trainable, Initializer::xavier);
	c = a->matmul(b, 2);
	*/
/*
	t = flux(tcr, { 8,4 }, tfloat, variable);
	t->randn(0, 1);
	a = flux(tcr, { 3,8 }, tfloat, variable);
	a->randn(0, 1);
	b = flux(tcr, { 4,3 }, tfloat, trainable, Initializer::xavier);
	c = a->matmul(b, 3);
	*/
	/*
	auto d = c->sigmoid();
	auto e = d->meanSquareError(t);

	op = gradient_descent_optimizer(tcr, 0.1)->minimize(e);
	
	float loss = 1000;
	for(intt i = 0;i < 100; i++) {
		tcr->run({ op });
		e->printo();
		if(loss < *(floatt *)e->begin_p()) {
			printf("!!! later big loss %f\n", *(floatt *)e->begin_p());
		}
		loss = *(floatt *)e->begin_p();
	}
	*/
/*
	a = flux(tcr, { 4,3 }, tfloat, variable);
	a->arange(-1);
	a->printo();
	b = flux(tcr, { 4,3,3 }, tfloat, variable);
	b->arange(-1);
	b->printo();
	c = flux(tcr, { 4,3 }, tfloat, variable);
	OneVar onevar;
	onevar.idxOne[0] = 1;//M
	onevar.idxOne[1] = 3;//K
	onevar.idxOne[2] = 3;//N
	onevar.idxOne[3] = 0;
	TENSOR(a->quantum)->mxData->mone(TRACER(tcr)->trcCxt(), TENSOR(b->quantum)->mxData,
		TENSOR(c->quantum)->mxData, 2, &onevar, AOP_MATMUL, 0, PDG, 0);
	c->printo();
	
	a = flux(tcr, { 2,2,3}, tfloat, variable);
	a->arange(-1);
	a->printo();
	//b = flux(tcr, { 2,2,3 }, tfloat, variable);
	//b->arange(-1);
	//b->printo();
	c = flux(tcr, { 2,2,3,3 }, tfloat, variable);
	OneVar onevar;
	onevar.idxOne[0] = 3;//M
	onevar.idxOne[1] = 1;//K
	onevar.idxOne[2] = 3;//N
	onevar.idxOne[3] = 0;
	TENSOR(a->quantum)->mxData->mone(TRACER(tcr)->trcCxt(), TENSOR(a->quantum)->mxData,
		TENSOR(c->quantum)->mxData, 2, &onevar, AOP_MATMUL, 0, PDG, 0);
	c->printo();
	*/
	delete tcr;
}
Flux *attention_dot2(Flux *inp, intt out_sz) //[batch, seq, feat]
{
	Flux *Q = inp->layer_dense(out_sz, ACTF_TANH, Initializer::xavier, "Q");
	Flux *K = inp->layer_dense(out_sz, ACTF_TANH, Initializer::xavier, "K");
	Flux *V = inp->layer_dense(out_sz, ACTF_TANH, Initializer::xavier, "V");

	Flux *cross_mul = Q->matmul(K, 2);
	Flux *scaled = *cross_mul / std::sqrt(K->fshape[K->fdim - 1]);
	Flux *attention_weight = scaled->softmax();
	return attention_weight->matmul(V);
}

void tt(void)
{
	Tracer *tcr = trace(0);

	tcr->modeset(1);
	//tcr->traceopt(20, -3);

	Flux *x_data = flux(tcr, "[[1, 2, 1, 1],\
          [2, 1, 3, 2],\
          [3, 1, 3, 4],\
          [4, 1, 5, 5],\
          [1, 7, 5, 5],\
          [1, 2, 5, 6],\
          [1, 6, 6, 6],\
          [1, 7, 7, 7]]", tfloat, variable);
	Flux *y_data = flux(tcr, "[[0, 0, 1],\
          [0, 0, 1],\
          [0, 0, 1],\
          [0, 1, 0],\
          [0, 1, 0],\
          [0, 1, 0],\
          [1, 0, 0],\
          [1, 0, 0]]", tfloat, variable);

	
	Flux *X = flux(tcr, { 8, 4 }, tfloat, variable);
	Flux *Y = flux(tcr, { 8, 3 }, tfloat, variable);

	intt nb_classes = 3;
	auto hypothesis = attention_dot2(X, nb_classes);
	hypothesis = hypothesis->softmax();
	auto cost = -1.0 * *(*hypothesis->log() * *Y)->mean();
	auto optimizer = gradient_descent_optimizer(tcr, 0.1)->minimize(cost);

	floatt loss = 1000;
	for(intt i = 0;i < 2000; i++) {
		X->feedf(x_data);
		Y->feedf(y_data);
		tcr->run({ optimizer });
		if(i % 200 == 0) cost->printo();
		if(loss < *(floatt *)cost->begin_p()) {
			printf("!!! later big loss %f\n", *(floatt *)cost->begin_p());
		}
		loss = *(floatt *)cost->begin_p();
	}
}
/*
void tt(void)
{
	Tracer *tcr = trace(0);

	tcr->modeset(1);

	Flux *x_data = flux(tcr, "[[1, 2, 1, 1],\
          [2, 1, 3, 2],\
          [3, 1, 3, 4],\
          [4, 1, 5, 5],\
          [1, 7, 5, 5],\
          [1, 2, 5, 6],\
          [1, 6, 6, 6],\
          [1, 7, 7, 7]]", tfloat, variable);
	Flux *y_data = flux(tcr, "[[0, 0, 1],\
          [0, 0, 1],\
          [0, 0, 1],\
          [0, 1, 0],\
          [0, 1, 0],\
          [0, 1, 0],\
          [1, 0, 0],\
          [1, 0, 0]]", tfloat, variable);

	Flux *X = flux(tcr, { -1, 4 }, tfloat, variable);
	Flux *Y = flux(tcr, { -1, 3 }, tfloat, variable);

	intt nb_classes = 3;

	Flux *W = flux(tcr, { 4, nb_classes }, tfloat, trainable);
	Flux *b = flux(tcr, { nb_classes }, tfloat, trainable);

	auto hypothesis = X->layer_dense(nb_classes, ACTF_TANH, Initializer::xavier, "Q")->softmax();

	auto cost = -1.0 * *(*hypothesis->log() * *Y)->mean();

	auto optimizer = gradient_descent_optimizer(tcr, 0.1)->minimize(cost);

	floatt loss = 1000;
	for(intt i = 0;i < 2000; i++) {
		X->feedf(x_data);
		Y->feedf(y_data);
		tcr->run({ optimizer });
		if(i % 200 == 0) cost->printo();
		if(loss < *(floatt *)cost->begin_p()) {
			printf("!!! later big loss %f\n", *(floatt *)cost->begin_p());
		}
		loss = *(floatt *)cost->begin_p();
	}
}*/
void ss(void)
{
	Tracer *tcr = trace(0);

	tcr->modeset(1);

	Flux *x_data = flux(tcr, "[[1, 2, 1, 1],\
          [2, 1, 3, 2],\
          [3, 1, 3, 4],\
          [4, 1, 5, 5],\
          [1, 7, 5, 5],\
          [1, 2, 5, 6],\
          [1, 6, 6, 6],\
          [1, 7, 7, 7]]", tfloat, variable);
	Flux *y_data = flux(tcr, "[[0, 0, 1],\
          [0, 0, 1],\
          [0, 0, 1],\
          [0, 1, 0],\
          [0, 1, 0],\
          [0, 1, 0],\
          [1, 0, 0],\
          [1, 0, 0]]", tfloat, variable);

	Flux *X = flux(tcr, { -1, 4 }, tfloat, variable);
	Flux *Y = flux(tcr, { -1, 3 }, tfloat, variable);

	intt nb_classes = 3;

	Flux *W = flux(tcr, { 4, nb_classes }, tfloat, trainable);
	Flux *b = flux(tcr, { nb_classes }, tfloat, trainable);

	auto hypothesis = (*X->dot(W, { {1}, {0} }) + *b)->softmax();

	auto cost = -1.0 * *(*hypothesis->log() * *Y)->mean();

	auto optimizer = gradient_descent_optimizer(tcr, 0.1)->minimize(cost);

	floatt loss = 1000;
	for(intt i = 0;i < 2000; i++) {
		X->feedf(x_data);
		Y->feedf(y_data);
		tcr->run({ optimizer });
		if(i % 200 == 0) cost->printo();
		if(loss < *(floatt *)cost->begin_p()) {
			printf("!!! later big loss %f\n", *(floatt *)cost->begin_p());
		}
		loss = *(floatt *)cost->begin_p();
	}
}
void soft_argmax_test(void)
{
	Tracer *tcr = trace(1);

	//tcr->modeset(1);

	Flux *x = flux(tcr, { 20, 1 }, tfloat, variable);
	Flux *y = flux(tcr, { 20 }, tfloat, variable);

	x->randn(0, 1);
	y->arange(-1);

	Flux *w = flux(tcr, { 1, 20 }, tfloat, trainable, Initializer::xavier);
	Flux *b = flux(tcr, { 20 }, tfloat, trainable, Initializer::xavier);
	w->randn(0, 1);
	b->randn(0, 1);
	auto c = *x->dot(w, { {1}, {0} }) + *b;
	printf("--------------- logit ---------------\n");
	c->printo(2, 10);
	c = c->prelu(0.2);
	//c = c->sigmoid();
	printf("--------------- sigmoid ---------------\n");
	c->printo(2, 10);
	Flux *amax = c->softmax();
	printf("--------------- softmax ---------------\n");
	amax->printo(2, 10);
	//auto tt = amax->log();
	//printf("--------------- log ---------------\n");
	//tt->printo(2, 10);

	auto pred = amax->argmax(-1);

	auto l = y->one_hot(20);
	l->printo(2, 10);

	auto cost = -1.0 * *(*amax->log() * *l)->mean();
	cost->printo();
	auto optimizer = gradient_descent_optimizer(tcr, 0.1)->minimize(cost);

	floatt loss = 1000, sloss;
	for(intt i = 0; i < 3000; i++) {
		tcr->run({ optimizer });
		cost->printo();
		if(loss < *(floatt *)cost->begin_p()) {
			printf("!!! later big loss %f\n", *(floatt *)cost->begin_p());
		}
		loss = *(floatt *)cost->begin_p();
		if(i == 0) sloss = loss;
	}
	tcr->run({ pred });
	printf("111 %f\n", sloss - loss);
	pred->printo(2, 10);
}
thrret thraa(thrarg arg)//mean square error test
{
	Tracer *tcr = trace(1), *tcr2;

	Flux *a, *b, *c, *w, *t, *op;
	float f;

	a = flux(tcr, { 512,4,3 }, tfloat, variable);
	a->arange(-1);
	//a->printo();
	w = flux(tcr, { 512,4,3 }, tfloat, trainable, nullx, "www");
	w->fill(0.8);
	//w->printo();
	b = a->mul(w);
	t = a->plus(0, 2);
	//t->printo();
	c = b->meanSquareError(t);

	op = gradient_descent_optimizer(tcr, 0.5)->minimize(c);

	tcr2 = trace(-1);
	tcr->portingGraph(tcr2);

	tcr2->init_train();

	op = tcr2->getFlux(op);

	while(1) {
		tcr2->run({ op });
		printf("%d\n", (intt)arg);
		//w->printo();
	}


	delete tcr;
}
#define X_TIME_SIZE 64
#define Y_TIME_SIZE X_TIME_SIZE
#define	FEATURE_SIZE 1
#define	HIDDEN_SIZE 32
#define	RNN_OUTPUT_SIZE 16

Tracer *sign_build(Flux *&op, Flux *&total_loss, Flux *&rnn_input, Flux *&rnn_output, Flux *&init_state,
	Flux *&cy_pred, Flux *&ccloss, Flux *&Why)
{
	Tracer *tcr = trace(0), *tcr2;
	Flux *Wxh, *Whh, *Wh, *bh, *by, *linear_w, *linear_b;

	tcr->lapset(1);

	rnn_input = flux(tcr, { 512, X_TIME_SIZE, FEATURE_SIZE }, tfloat, variable);
	rnn_output = flux(tcr, { 512, X_TIME_SIZE, FEATURE_SIZE }, tfloat, variable);
	init_state = flux(tcr, { 512, HIDDEN_SIZE }, tfloat, variable);

	rnn_input->randn(0, 0.5);
	rnn_output->randn(0, 0.5);
	init_state->fill((floatt)0);

	Wxh = flux(tcr, { FEATURE_SIZE, HIDDEN_SIZE }, tfloat, trainable, Initializer::xavier);
	Whh = flux(tcr, { HIDDEN_SIZE, HIDDEN_SIZE }, tfloat, trainable, Initializer::xavier);
	//Wh = concat({ Wxh, Whh }, 0);
	Why = flux(tcr, { HIDDEN_SIZE, RNN_OUTPUT_SIZE }, tfloat, trainable, Initializer::xavier);
	bh = flux(tcr, { 1, HIDDEN_SIZE }, tfloat, trainable, Initializer::xavier);
	by = flux(tcr, { FEATURE_SIZE, 1 }, tfloat, trainable, Initializer::xavier);


	linear_w = flux(tcr, { RNN_OUTPUT_SIZE, FEATURE_SIZE }, tfloat, trainable, Initializer::xavier);
	linear_b = flux(tcr, { FEATURE_SIZE }, tfloat, trainable, Initializer::xavier);

	vector<Flux *> y_preds;
	vector<Flux *> losses;
	Flux *hiddens[X_TIME_SIZE + 1];
	memset(hiddens, 0x00, sizeof((X_TIME_SIZE + 1) * sizeof(Flux *)));
	hiddens[X_TIME_SIZE] = init_state;

	auto unstacked_inputs = rnn_input->unstack(1);
	auto unstacked_outputs = rnn_output->unstack(1);

	Flux *input_t, *y_true, *concat_x, *hidden, *y_pred, *loss, *a, *b, *c, *d;

	for(intt t = 0;t < unstacked_inputs->size(); t++) {
		input_t = unstacked_inputs->at(t);
		y_true = unstacked_outputs->at(t);
		//concat_x = tf.concat([input_t, hiddens[t - 1]], axis = 1);
		//hidden = tf.tanh(tf.matmul(concat_x, Wh) + bh);
		a = input_t->dot(Wxh, { {1}, {0} });
		hidden = (t == 0 ? hiddens[X_TIME_SIZE] : hiddens[t - 1]);
		b = hidden->dot(Whh, { {1}, {0} });
		c = a->plus(b);
		d = c->plus(bh);
		hidden = d->tanh();
		y_pred = hidden->dot(Why, { {1}, {0} });
		y_pred = y_pred->plus(by);
		y_pred = y_pred->tanh();

		y_pred = y_pred->dot(linear_w, { {1}, {0} });
		y_pred = y_pred->plus(linear_b);

		loss = y_pred->meanSquareError(y_true);
		hiddens[t] = hidden;
		y_preds.push_back(y_pred);
		losses.push_back(loss);
	}
	cy_pred = concat(&y_preds, 1);
	ccloss = concat(&losses, 0);
	total_loss = ccloss->mean();
	op = gradient_descent_optimizer(tcr)->minimize(total_loss);
	//tcr->init_train();
	/*
	tcr2 = trace(-1);
	tcr->portingGraph(tcr2);
	tcr2->sizeBatch(10000);

	tcr2->init_train();

	rnn_input = tcr2->getFlux(rnn_input);
	rnn_output = tcr2->getFlux(rnn_output);
	init_state = tcr2->getFlux(init_state);
	op = tcr2->getFlux(op);
	total_loss = tcr2->getFlux(total_loss);
	return tcr2;
	*/
	return tcr;
}

Tracer *sign_build2(Flux *&op, Flux *&total_loss, Flux *&rnn_input, Flux *&rnn_output, Flux *&init_state,
	Flux *&cy_pred, Flux *&ccloss, Flux *&Why, Flux *&linear_w, Flux *&linear_b)
{
	Tracer *tcr = trace(1, "00"), *tcr2;
	Flux *Wxh, *Whh, *Wh, *bh, *by;

	//tcr->lapset(1);

	rnn_input = flux(tcr, { -1, X_TIME_SIZE, FEATURE_SIZE }, tfloat, variable);
	rnn_output = flux(tcr, { -1, X_TIME_SIZE, FEATURE_SIZE }, tfloat, variable);
	init_state = flux(tcr, { -1, HIDDEN_SIZE }, tfloat, variable);

	rnn_input->randn(0, 0.5);
	rnn_output->randn(0, 0.5);
	init_state->fill((floatt)0);

	Wxh = flux(tcr, { FEATURE_SIZE, HIDDEN_SIZE }, tfloat, trainable, Initializer::xavier, "aa");
	Whh = flux(tcr, { HIDDEN_SIZE, HIDDEN_SIZE }, tfloat, trainable, Initializer::xavier, "bb");
	Wh = concat({ Wxh, Whh }, 0);
	Why = flux(tcr, { HIDDEN_SIZE, RNN_OUTPUT_SIZE }, tfloat, trainable, Initializer::xavier);
	bh = flux(tcr, { 1, HIDDEN_SIZE }, tfloat, trainable, Initializer::xavier);
	by = flux(tcr, { FEATURE_SIZE, 1 }, tfloat, trainable, Initializer::xavier);


	linear_w = flux(tcr, { RNN_OUTPUT_SIZE, FEATURE_SIZE }, tfloat, trainable, Initializer::xavier, "cc");
	linear_b = flux(tcr, { FEATURE_SIZE }, tfloat, trainable, Initializer::xavier, "dd");

	vector<Flux *> y_preds;
	vector<Flux *> losses;
	Flux *hiddens[X_TIME_SIZE + 1];
	memset(hiddens, 0x00, sizeof((X_TIME_SIZE + 1) * sizeof(Flux *)));
	hiddens[X_TIME_SIZE] = init_state;

	//Wh = Wh->reshape({ Wh->fshape[0], 4, HIDDEN_SIZE / 4  });
	//Wh = Wh->reshape({ Wh->fshape[0], HIDDEN_SIZE });
	auto rnn_input2 = rnn_input->bypass();

	auto unstacked_inputs = rnn_input2->unstack(1);
	auto unstacked_outputs = rnn_output->unstack(1);

	Flux *input_t, *y_true, *concat_x, *hidden, *y_pred, *loss, *a, *b, *c, *d;

	for(intt t = 0;t < unstacked_inputs->size(); t++) {
		input_t = unstacked_inputs->at(t);
		y_true = unstacked_outputs->at(t);
		hidden = (t == 0 ? hiddens[X_TIME_SIZE] : hiddens[t - 1]);
		concat_x = concat({ input_t, hidden }, 1);
		a = concat_x->dot(Wh, { {1}, {0} });
		b = a->plus(bh);
		hidden = b->tanh();
		y_pred = hidden->dot(Why, { {1}, {0} });
		y_pred = y_pred->plus(by);
		y_pred = y_pred->tanh();

		y_pred = y_pred->dot(linear_w, { {1}, {0} });
		y_pred = y_pred->plus(linear_b);

		loss = y_pred->meanSquareError(y_true);
		hiddens[t] = hidden;
		y_preds.push_back(y_pred);
		losses.push_back(loss);
	}
	cy_pred = concat(&y_preds, 1);
	ccloss = concat(&losses, 0);
	total_loss = ccloss->mean();
	op = gradient_descent_optimizer(tcr)->minimize(total_loss);
	//tcr->init_train();

	return tcr;
}
/*
thrret thr_sign(thrarg arg)//mean square error test
{
	Flux *total_loss, *op, *cy_pred, *ccloss, *Why, *rnn_input, *rnn_output, *init_state;
	Tracer *tcr = sign_build(op, total_loss, rnn_input, rnn_output, init_state, cy_pred, ccloss, Why);
	//Why->printo();
	while(1) {
		tcr->run({ op, total_loss });
		//Why->printo();
		//cy_pred->printo();
		//ccloss->printo();
		printf("%d: ", (intt)arg);
		total_loss->printo();
	}

	delete tcr;
}
void ll(void)
{
	xthread_create((void *)thr_sign, (void *)1);
	xthread_create((void *)thr_sign, (void *)2);
	xthread_create((void *)thr_sign, (void *)3);
	thr_sign((void *)4);
}*/
#define BATCH_SZ 16
void graph_exec_test(void)
{
	Flux *total_loss, *op, *cy_pred, *ccloss, *Why, *rnn_input, *rnn_output, *init_state, *linear_w, *linear_b;
	Tracer *tcr = sign_build2(op, total_loss, rnn_input, rnn_output, init_state, cy_pred, ccloss, Why, linear_w, linear_b);
	//Why->printo();
	Tracer *tcr2 = trace(1);
	Flux *sample_x, *sample_y, *state;
	sample_x = flux(tcr2, { BATCH_SZ, X_TIME_SIZE, FEATURE_SIZE }, tfloat, trainable);
	sample_y = flux(tcr2, { BATCH_SZ, X_TIME_SIZE, FEATURE_SIZE }, tfloat, trainable);
	state = flux(tcr2, { BATCH_SZ, HIDDEN_SIZE }, tfloat, trainable);

	sample_x->randn(0, 0.5);
	sample_y->randn(0, 0.5);
	state->fill((floatt)0);
	//sample_x->printo();
	//sample_y->printo();
	//state->printo();
	//tcr2->saveWeight();
	//tcr->gprset(0.1);
	//tcr->modeset(1);
	tcr->sizeBatch(4);
	unit lap;
	float loss = 1000;
	for(intt i = 0;i < 100; i++) {
		rnn_input->feedf(sample_x);
		rnn_output->feedf(sample_y);
		init_state->feedf(state);
		lap = xucurrenttime();
		tcr->run({ op, total_loss });
		printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
		//Why->printo();
		//cy_pred->printo();
		//linear_w->printg();
		//linear_b->printg();
		//ccloss->printo();
		total_loss->printo();
		if(loss < *(floatt *)total_loss->begin_p()) {
			printf("!!! later big loss %f\n", *(floatt *)total_loss->begin_p());
		}
		loss = *(floatt *)total_loss->begin_p();
	}
	Flux *test_x = flux(tcr, { 1, X_TIME_SIZE, FEATURE_SIZE }, tfloat, variable);
	test_x->randn(0, 0.5);
	for(intt i = 0;i < 3; i++) {
		rnn_input->feedf(test_x);
		rnn_output->feedf(test_x);
		init_state->feedf(state);
		lap = xucurrenttime();
		tcr->run({ total_loss, cy_pred });
		printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
		cy_pred->printo();
		total_loss->printo();
	}
	delete tcr;
}
void weight_save_load_test(void)
{
	Flux *total_loss, *op, *cy_pred, *ccloss, *Why, *rnn_input, *rnn_output, *init_state, *linear_w, *linear_b;
	Tracer *tcr = sign_build2(op, total_loss, rnn_input, rnn_output, init_state, cy_pred, ccloss, Why, linear_w, linear_b);
	//Why->printo();
	Tracer *tcr2 = trace(1, "11");
	Flux *sample_x, *sample_y, *state;
	sample_x = flux(tcr2, { BATCH_SZ, X_TIME_SIZE, FEATURE_SIZE }, tfloat, trainable);
	sample_y = flux(tcr2, { BATCH_SZ, X_TIME_SIZE, FEATURE_SIZE }, tfloat, trainable);
	state = flux(tcr2, { BATCH_SZ, HIDDEN_SIZE }, tfloat, trainable);

	tcr2->loadWeight();
	//sample_x->randn(0, 0.5);
	//sample_y->randn(0, 0.5);
	//state->fill((floatt)0);
	//sample_x->printo();
	//sample_y->printo();
	//state->printo();
	//tcr2->saveWeight();
	//tcr->gprset(0.1);
	//tcr->modeset(1);
	tcr->sizeBatch(4);
	unit lap;
	float loss = 1000;
	for(intt i = 0;i < 100; i++) {
		rnn_input->feedf(sample_x);
		rnn_output->feedf(sample_y);
		init_state->feedf(state);
		lap = xucurrenttime();
		tcr->run({ op, total_loss });
		printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
		//Why->printo();
		//cy_pred->printo();
		//linear_w->printg();
		//linear_b->printg();
		//ccloss->printo();
		total_loss->printo();
		if(loss < *(floatt *)total_loss->begin_p()) {
			printf("!!! later big loss %f\n", *(floatt *)total_loss->begin_p());
		}
		loss = *(floatt *)total_loss->begin_p();
	}
	//tcr->saveWeight();
	Flux *test_x = flux(tcr, { 1, X_TIME_SIZE, FEATURE_SIZE }, tfloat, variable);
	test_x->randn(0, 0.5);
	for(intt i = 0;i < 3; i++) {
		rnn_input->feedf(test_x);
		rnn_output->feedf(test_x);
		init_state->feedf(state);
		lap = xucurrenttime();
		tcr->run({ total_loss, cy_pred });
		printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
		cy_pred->printo();
		total_loss->printo();
	}
	delete tcr;
}
void embedding_onehot_graph_exec_test(void)
{
	Tracer *tcr = trace(1);
	Flux *a, *b, *c, *w, *t;

	a = flux(tcr, "[[[1, 3, 5],\
		[3, 5, 2]],\
		[[5, 0, 3],\
		[1, 3, 5]],\
		[[3, 5, 1],\
		[5, 4, 3]]]");

	t = a->one_hot(6);
	t->shape();
	t->printo();
	Flux *e1 = flux(tcr, { 6, 6 }, tfloat, trainable, Initializer::xavier, "embed_lookup");

	auto z = flux(tcr, { 1, 6 }, tfloat, variable);
	printf("yyy: %p\n", z);
	z->fill(0.0);
	auto e = concat({ z, e1->slice({{1, -1}, {}}) }, 0);

	auto *o = e->embedding_lookup(a);
	o->shape();
	o->printo();
	//b = a->argmax(2);
	//b->printo();
	w = flux(tcr, { 6, 6 }, tfloat, trainable);
	//b = a->mul(w);
	b = o->dot(w, { { 3 }, { 0 } });
	//b->argmax(-1)->printo();
	c = b->softmaxCrossEntropy(t)->mean();
	auto op = adam_optimizer(tcr)->minimize(c);
	for(intt i = 0;i < 100; i++) {
		tcr->run({ op, c });
		c->printo();
		//o->printo();
		//e1->printo();
	}
	b->printo();
	b->argmax(-1)->printo();
	w->printo();
}
void cnet_not_embedding_mean_square_test(void)
{
	Flux *total_loss, *op, *cy_pred, *ccloss, *Why, *rnn_input, *rnn_output, *init_state;
	Tracer *tcr = trace(0);

	rnn_input = flux(tcr, { -1, X_TIME_SIZE, FEATURE_SIZE }, tfloat, variable);
	rnn_output = flux(tcr, { -1, X_TIME_SIZE, FEATURE_SIZE }, tfloat, variable);
	rnn_input->arange(-1);
	Generic cell(rnn_input, rnn_output, 32, 0, 0, 0);

	Tracer *tcr2 = trace(1);
	//tcr2->modeset(-1);
	Flux *sample_x, *sample_y;
	sample_x = flux(tcr2, { BATCH_SZ, X_TIME_SIZE, FEATURE_SIZE }, tfloat, variable);
	sample_y = flux(tcr2, { BATCH_SZ, X_TIME_SIZE, FEATURE_SIZE }, tfloat, variable);

	sample_x->randn(0, 0.5);
	sample_y->randn(0, 0.5);
	//sample_y->printo();
	tcr->sizeBatch(4);//에러발생, 나중에 디버깅
	//tcr->gprset(0.1);
	//tcr->modeset(1);
	unit lap;
	float loss = 1000;
	for(intt i = 0;i < 10; i++) {
		rnn_input->feedf(sample_x);
		rnn_output->feedf(sample_y);
		lap = xucurrenttime();
		total_loss = cell.train();
		printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
		//Why->printo();
		//cy_pred->printo();
		//ccloss->printo();
		total_loss->printo();
		if(loss < *(floatt *)total_loss->begin_p()) {
			printf("!!! later big loss %f\n", *(floatt *)total_loss->begin_p());
		}
		loss = *(floatt *)total_loss->begin_p();
	}
	Flux *test_x = flux(tcr, { 1, X_TIME_SIZE, FEATURE_SIZE }, tfloat, variable);
	test_x->randn(0, 0.5);
	for(intt i = 0;i < 3; i++) {
		rnn_input->feedf(test_x);
		rnn_output->feedf(test_x);
		lap = xucurrenttime();
		cy_pred = cell.predict(0);
		printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
		cy_pred->printo();
		cell.closs->printo();
	}
	delete tcr;
}
void cnet_embedding_cross_entropy_test(void)
{
	Tracer *tcr = trace(1, "kkk");
	Flux *a, *a2, *b, *c, *w, *t;

	//tcr->truncWeight();
	//tcr->modeset(1);
	Flux *aa = flux(tcr, { 10, 16, 1 }, tfloat, variable);
	aa->arange(-1);

	//TRACER(tcr)->strideJump = 4;
	a = flux(tcr, { -1, 16, 1 }, tfloat, variable);
	//a2 = flux(tcr, { -1, 16, 1 }, tfloat, variable);
	//Generic cell(a, a2, 16, 10 * 16, 10 * 16, 8);
	Generic cell(a, a, 16, 10 * 16, 10 * 16, 8);

	//tcr->sizeBatch(4);
	for(intt i = 0;i < 30; i++) {
		a->feedf(aa);
		//a2->feedf(aa);
		b = cell.train();
		b->printo();
	}
	c = cell.predict(0);
	c->printo();
	tcr->saveWeight();
}
void cnet_embedding_cross_entropy_test2(void)
{
	Tracer *tcr = trace(1, "kkk");
	Flux *a, *a2, *b, *c, *w, *t;

	//tcr->truncWeight();
	//tcr->modeset(1);
	Flux *aa = flux(tcr, { 10, 29, 1 }, tfloat, variable);
	aa->arange(-1);
	Flux *bb = flux(tcr, { 10, 5, 1 }, tfloat, variable);
	bb->arange(-1);

	TRACER(tcr)->traceopt(0, 2);
	a = flux(tcr, { -1, 29, 1 }, tfloat, variable);
	a2 = flux(tcr, { -1, 5, 1 }, tfloat, variable);
	Generic cell(a, a2, 16, -1, 10 * 5, 0);

	//tcr->sizeBatch(4);
	for(intt i = 0; i < 30; i++) {
		a->feedf(aa);
		a2->feedf(bb);
		b = cell.train();
		b->printo();
	}
	c = cell.predict(0);
	c->printo();
	tcr->saveWeight();
}
void cnet_embedding_cross_entropy_no_feedf_test(void)
{
	Tracer *tcr = trace(1, "kkk");
	Flux *a, *b, *c, *w, *t;

	//tcr->truncWeight();

	//a = flux(tcr, { 2, 16, 1 }, tfloat, variable);
	//a->arange(-1);
	//Generic cell(a, a, 16, 2 * 16, 2 * 16, 8);
	//tcr->modeset(1);
	a = flux(tcr, { 10, 16, 1 }, tfloat, variable);
	a->arange(-1);
	Generic cell(a, a, 16, 10 * 16, 10 * 16, 8);

	for(intt i = 0;i < 10; i++) {
		b = cell.train();
		b->printo();
	}
	c = cell.predict(0);
	c->printo();
	tcr->saveWeight();
}
void cnet_not_embedding_mean_square_convolution_test(void)
{
	Flux *total_loss, *op, *cy_pred, *ccloss, *Why, *rnn_input, *rnn_output, *init_state;
	Tracer *tcr = trace(0);

	rnn_input = flux(tcr, { -1, 64, 1 }, tfloat, variable);
	rnn_output = flux(tcr, { -1, 64, 1 }, tfloat, variable);
	rnn_input->arange(-1);

	tcr->npset(200);
	TRACER(tcr)->traceopt(0, 4);
	//TRACER(tcr)->traceopt(15, 1);//layer normal
	TRACER(tcr)->traceopt(51, 1);
	Generic cell(rnn_input, rnn_output, 32, 0, 0, 0);
	/*Generic cell(rnn_input, 32, 32, 0, 0);
	cell.reduction(8);
	cell.decompose(32);
	cell.decompose(64);
	cell.connect(rnn_output, 0, 0);*/

	Tracer *tcr2 = trace(1);
	//tcr2->modeset(1);
	Flux *sample_x, *sample_y;
	sample_x = flux(tcr2, { 10, 64, 1 }, tfloat, variable);
	sample_y = flux(tcr2, { 10, 64, 1 }, tfloat, variable);

	sample_x->randn(0, 0.5);
	sample_y->randn(0, 0.5);
	//sample_y->printo();
	//tcr->sizeBatch(4);
	//tcr->gprset(0.1);
	//tcr->modeset(1);
	unit lap;
	float loss = 1000;
	Flux *batloss;
	for(intt i = 0;i < 100; i++) {
		rnn_input->feedf(sample_x);
		rnn_output->feedf(sample_y);
		lap = xucurrenttime();
		total_loss = cell.train();
		printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
		//Why->printo();
		//cy_pred->printo();
		//ccloss->printo();
		total_loss->printo();
		if(loss < *(floatt *)total_loss->begin_p()) {
			printf("!!! later big loss %f\n", *(floatt *)total_loss->begin_p());
		}
		loss = *(floatt *)total_loss->begin_p();
		cell.predict(&batloss);
		batloss->printo();
	}
	Flux *test_x = flux(tcr, { 1, 64, 1 }, tfloat, variable);
	test_x->randn(0, 0.5);
	for(intt i = 0;i < 3; i++) {
		rnn_input->feedf(test_x);
		rnn_output->feedf(test_x);
		lap = xucurrenttime();
		cy_pred = cell.predict(0);
		printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
		cy_pred->printo();
		cell.closs->printo();
	}
	delete tcr;
}
void cnet_embedding_cross_entropy_convolution_test(intt sz)
{
	Tracer *tcr = trace(0);
	Flux *a, *a2, *b, *c, *w, *t, *aa, *ax, *ay;

	//tcr->modeset(1);
	/*
	a = flux(tcr, { 2, 3, 3 }, tfloat, variable);
	a->arange(-1);
	a->printo();
	a2 = flux(tcr, { 2, 3, 3 }, tfloat, variable);
	a2->arange(-1);
	a2->printo();
	c = flux(tcr, { 2, 3 }, tfloat, variable);
	TENSOR(a->quantum)->mxData->mdiag_mul(nullx, TENSOR(a2->quantum)->mxData, TENSOR(c->quantum)->mxData);
	c->printo();
	TENSOR(c->quantum)->mxData->mdiag_fill(nullx, TENSOR(a2->quantum)->mxData);
	a2->printo();

	c = flux(tcr, { 2, 3 }, tfloat, variable);
	c->arange(-1);
	c = *c * -1.0;
	TENSOR(c->quantum)->mxData->mdiag_fill(nullx, TENSOR(a2->quantum)->mxData);
	a2->printo();
	*/
	/*
	a = flux(tcr, { 5 }, tfloat, variable);
	a->arange(-1);
	a = a->expand_dims(1);
	a->printo();
	a2 = flux(tcr, { 4 }, tfloat, variable);
	a2->arange(-1);
	a2 = a2->expand_dims(0);
	a2->printo();
	c = a->dot(a2, { {1}, {0} });
	c->printo();*/
	/*
	ax = flux(tcr, { 1, 8 }, tfloat, variable);
	ax->expofill(2);
	ax->printo();
	ax->shape();
	ax = ax->expand_elen(10, 2);
	ax->printo();
	ax->shape();
	ax = ax->expand_elen(3, 0);
	ax->printo();
	ax->shape();*/
	//TRACER(tcr)->npset(300);
	//tcr->truncWeight();
	/*
	aa = flux(tcr, { 1, 8, 1 }, tfloat, variable);
	aa->arange(-1);
	aa = rsc::rsc_combination(aa, 8, 8, 1, 1, 0, 1);
	aa->printo(1, 2);
	printf("------------------------------------------------ 1111 \n");
	
	aa = flux(tcr, { 1, 16, 1 }, tfloat, variable);
	aa->arange(-1);
	aa = rsc::rsc_combination(aa, 8, 6, 1, 3, 0, 1);
	aa->printo(1, 2);
	printf("------------------------------------------------ 2222 \n");
	
	aa = flux(tcr, { 1, 32, 1 }, tfloat, variable);
	aa->arange(-1);
	aa = rsc::rsc_combination(aa, 8, 6, 1, 1, 0, 1);
	aa->printo(1, 2);
	printf("------------------------------------------------ 3333 \n");

	aa = flux(tcr, { 1, 8, 1 }, tfloat, variable);
	aa->arange(-1);
	aa = rsc::rsc_combination(aa, 8, 8, 1, 3, 0, 1);
	aa->printo(1, 2);
	printf("------------------------------------------------ 4444 \n");
	aa = flux(tcr, { 1, 16, 1 }, tfloat, variable);
	aa->arange(-1);
	aa = rsc::rsc_combination(aa, 8, 6, 1, 3, 0, 1);
	aa->printo(1, 2);
	printf("------------------------------------------------ 5555 \n");
	aa = flux(tcr, { 1, 32, 1 }, tfloat, variable);
	aa->arange(-1);
	aa = rsc::rsc_combination(aa, 8, 6, 1, 3, 0, 1);
	aa->printo(1, 2);
	printf("------------------------------------------------ 6666 \n");
	*/
	/*
	aa = flux(tcr, { 1, 8, 1 }, tfloat, variable);
	aa->arange(-1);
	aa = rsc::rsc_combination(aa, 8, 4, 1, 1, 0, 1);
	aa->printo(1, 2);
	printf("------------------------------------------------ 7777 \n");
	
	aa = flux(tcr, { 1, 16, 1 }, tfloat, variable);
	aa->arange(-1);
	aa = rsc::rsc_combination(aa, 8, 4, 1, 1, 0, 1);
	aa->printo(1, 2);
	printf("------------------------------------------------ 8888 \n");
	aa = flux(tcr, { 1, 32, 1 }, tfloat, variable);
	aa->arange(-1);
	aa = rsc::rsc_combination(aa, 8, 4, 1, 1, 0, 1);
	aa->printo(1, 2);
	printf("------------------------------------------------ 9999 \n");

	aa = flux(tcr, { 1, 8, 1 }, tfloat, variable);
	aa->arange(-1);
	aa = rsc::rsc_combination(aa, 8, 4, 1, 3, 0, 1);
	aa->printo(1, 2);
	printf("------------------------------------------------ aaaa \n");
	aa = flux(tcr, { 1, 16, 1 }, tfloat, variable);
	aa->arange(-1);
	aa = rsc::rsc_combination(aa, 8, 4, 1, 3, 0, 1);
	aa->printo(1, 2);
	printf("------------------------------------------------ bbbb \n");
	aa = flux(tcr, { 1, 32, 1 }, tfloat, variable);
	aa->arange(-1);
	aa = rsc::rsc_combination(aa, 8, 4, 1, 3, 0, 1);
	aa->printo(1, 2);
	printf("------------------------------------------------ cccc \n");

	aa = flux(tcr, { 1, 8, 1 }, tfloat, variable);
	aa->arange(-1);
	aa = rsc::rsc_combination(aa, 8, 3, 1, 1, 0, 1);
	aa->printo(1, 2);
	printf("------------------------------------------------ dddd \n");
	aa = flux(tcr, { 1, 16, 1 }, tfloat, variable);
	aa->arange(-1);
	aa = rsc::rsc_combination(aa, 8, 3, 1, 1, 0, 1);
	aa->printo(1, 2);
	printf("------------------------------------------------ eeee \n");
	aa = flux(tcr, { 1, 32, 1 }, tfloat, variable);
	aa->arange(-1);
	aa = rsc::rsc_combination(aa, 8, 3, 1, 1, 0, 1);
	aa->printo(1, 2);
	printf("------------------------------------------------ ffff \n");

	aa = flux(tcr, { 1, 8, 1 }, tfloat, variable);
	aa->arange(-1);
	aa = rsc::rsc_combination(aa, 8, 3, 1, 3, 0, 1);
	aa->printo(1, 2);
	printf("------------------------------------------------ gggg \n");
	aa = flux(tcr, { 1, 16, 1 }, tfloat, variable);
	aa->arange(-1);
	aa = rsc::rsc_combination(aa, 8, 3, 1, 3, 0, 1);
	aa->printo(1, 2);
	printf("------------------------------------------------ hhhh \n");
	aa = flux(tcr, { 1, 32, 1 }, tfloat, variable);
	aa->arange(-1);
	aa = rsc::rsc_combination(aa, 8, 3, 1, 3, 0, 1);
	aa->printo(1, 2);
	printf("------------------------------------------------ iiii \n");
	*/
/*
	a = flux(tcr, { -1, 8, 1 }, tfloat, variable);
	a2 = flux(tcr, { -1, 8, 1 }, tfloat, variable);
	a->arange(-1);
	TRACER(tcr)->zeroPadding = 3;//3,4
	TRACER(tcr)->ebatch_t = 1;//1,2
	TRACER(tcr)->strideJump = 6;//4,3,6
	Generic cell2(a, a2, 16, 0, 0, 0);
	
	a = flux(tcr, { -1, 16, 1 }, tfloat, variable);
	a2 = flux(tcr, { -1, 16, 1 }, tfloat, variable);
	a->arange(-1);
	TRACER(tcr)->zeroPadding = 3;//3,4
	TRACER(tcr)->ebatch_t = 2;//1,2
	TRACER(tcr)->strideJump = 4;//4
	Generic cell2(a, a2, 16, 0, 0, 0);

	a = flux(tcr, { -1, 32, 1 }, tfloat, variable);
	a2 = flux(tcr, { -1, 32, 1 }, tfloat, variable);
	a->arange(-1);
	TRACER(tcr)->zeroPadding = 3;//3,4
	TRACER(tcr)->ebatch_t = 2;//1,2
	TRACER(tcr)->strideJump = 6;//6
	Generic cell2(a, a2, 16, 0, 0, 0);
	*/
//tcr->lapset(1);
	intt x_seq_len = 64;//1//56;//32;
	intt y_seq_len = 64;//1;//7;//32;
	intt batch_sz = 16;
	ax = flux(tcr, { batch_sz, x_seq_len, 1 }, tfloat, variable);
	ax->arange(-1);
	ay = flux(tcr, { batch_sz, y_seq_len, 1 }, tfloat, variable);
	ay->arange(-1);
	//ax->printo(1, 0);
	//tcr->sizeBatch(4);
	TRACER(tcr)->traceopt(1, 1);
	TRACER(tcr)->traceopt(0, 4);//8
	//TRACER(tcr)->ebatch_t = 1;
	//TRACER(tcr)->strideJump = 3;
	//TRACER(tcr)->traceopt(6, 1);
	//TRACER(tcr)->traceopt(9, 2);
	//TRACER(tcr)->traceopt(5, 0.5);
	TRACER(tcr)->traceopt(8, 1);//small block
	//TRACER(tcr)->traceopt(10, 1);
	TRACER(tcr)->traceopt(11, 1);
	TRACER(tcr)->traceopt(9, 0);
	//TRACER(tcr)->traceopt(13, 0.25);
	//TRACER(tcr)->traceopt(16, 2);
	//TRACER(tcr)->traceopt(14, 40);
	//TRACER(tcr)->traceopt(15, 1);//layer normal
	//TRACER(tcr)->traceopt(21, 1);//어텐션 수행
	//TRACER(tcr)->traceopt(16, 3);//최종 인코딩에서만 어텐션 수행
	//TRACER(tcr)->traceopt(12, 64);
	//TRACER(tcr)->traceopt(17, 1);
	//TRACER(tcr)->traceopt(18, 100);//prelu
	//TRACER(tcr)->traceopt(20, 3);//by pass print
	//TRACER(tcr)->traceopt(22, 1);//optimizier
	//TRACER(tcr)->traceopt(23, 0.1);//learnning rate
	//TRACER(tcr)->traceopt(7, 1);
	//TRACER(tcr)->traceopt(51, 1);
	TRACER(tcr)->traceopt(64, 4);
	a = flux(tcr, { -1, x_seq_len, 1 }, tfloat, variable);
	a->arange(-1);
	a2 = flux(tcr, { -1, y_seq_len, 1 }, tfloat, variable);
//a->arange(-1);
	//TRACER(tcr)->traceopt(25, 1);
	//TRACER(tcr)->traceopt(4, 16);
	Generic cell(a, a2, 32, batch_sz * x_seq_len, batch_sz * y_seq_len, 8 , ACTF2_PRELU);
	/*Generic cell(a, 16, 16, 10 * seq_len, 8);//10 * seq_len (vocab size)
	cell.reduction(8);
	cell.decompose(16);
	cell.decompose(32);
	cell.connect(a2, 10 * 32, 0);*/
	//tcr->saveWeight();
	//tcr->sizeBatch(4);
	if(sz) tcr->traceopt(8, 1);
	Flux *loss;
	floatt sloss;
	for(intt i = 0;i < 50; i++) {
		//tcr->printWeight();
		a->feedf(ax);
		a2->feedf(ay);
		b = cell.train();
		printf("\n");
		b->printo();
		printf("\n");
		if(i == 0) sloss = b->at_d(0);
		if(sz > 0 && i % 3 == 0) {
			printf((bytet *)"----++--\n");
			a->feedf(ax);
			c = cell.predict(&loss);
			loss->printo();
			//c->printo();
			//if(sz > 1) tcr->printWeight();
		}
	}
	printf("loss down: %f\n", sloss - b->at_d(0));
	//tcr->saveWeight();
	//c = cell.predict(&loss);
	//c->printo();
	//loss->printo();
}
/*
void make_seq_data(Tracer *tcr, Flux *dat, intt seq_len, Flux *&in, Flux *&tar)
{
	Flux *az = flux(tcr, { dat->fshape[0], 1, dat->fshape[2] }, tfloat, variable);
	az->fill(0.0);
	in = flux(tcr, { dat->fshape[0], seq_len, dat->fshape[2] }, tfloat, variable);
	in->fill(0.0);
	tar = flux(tcr, { dat->fshape[0], seq_len, dat->fshape[2] }, tfloat, variable);
	tar->fill(0.0);

	in = concat({ az, dat }, 1);
	in = concat({ dat, az }, 1);
	//in->copyf(az->begin_p(), az->sizef());
	//in->copyf(dat->begin_p(), dat->sizef(), az->sizef());

	
	tar->copyf(dat->begin_p(), dat->sizef());
	tar->copyf(az->begin_p(), az->sizef(), dat->sizef());
}*/
void cnet_embedding_cross_entropy_dual_test(intt sz)
{
	Trace *tcr = (Trace *)trace(1);
	Flux *in_gate, *tar_gate, *by, *by_gate, *seq_d, *b = nullx, *ax, *ay, *az, *c;

	//tcr->modeset(1);
	tcr->npset(10000);
	//tcr->setgpudev(1);

	intt x_seq_len = 32;
	intt y_seq_len = 32;
	intt batch_sz = 8;
	intt half_len = x_seq_len / 2 - 1;

	by = flux(tcr, { batch_sz, y_seq_len, 1 }, tfloat, variable);

	az = flux(tcr, { batch_sz, 1, 1 }, tfloat, variable);//token
	az->fill(0.0);

	seq_d = flux(tcr, { batch_sz, half_len, 1 }, tfloat, variable);//input data
	seq_d->arange(-1);

	floatt ii = batch_sz * half_len;
	auto seq_y = *seq_d + ii;
	//seq_y->printo();

	by->howrite(seq_d);//input
	//by->printo(2);
	by->howrite(az, half_len);//padding
	//by->printo(2);
	by->howrite(az, half_len + 1);//go token
	by->howrite(seq_d, half_len + 2);//target
	
	by->printo(2, 3);

	intt in_plus = 0;
	if(in_plus) {
		ay = flux(tcr, { batch_sz, x_seq_len, 1 }, tfloat, variable);
		ay->howrite(seq_d);//input
		ay->howrite(az, half_len);//padding
		ay->howrite(seq_y, half_len + 1);//target
		ay->howrite(az, half_len + half_len + 1);//end token
	} else {
		ay = flux(tcr, { batch_sz, half_len + 1, 1 }, tfloat, variable);
		ay->howrite(seq_y);//target
		ay->howrite(az, half_len);//padding
	}
	ay->printo(2, 5);
	//ax->printo(1, 0);
	//tcr->sizeBatch(4);
	TRACER(tcr)->traceopt(1, 1);
	//TRACER(tcr)->traceopt(0, 4);//8
	//TRACER(tcr)->ebatch_t = 1;
	//TRACER(tcr)->strideJump = 3;
	//TRACER(tcr)->traceopt(6, 1);
	//TRACER(tcr)->traceopt(9, 2);
	//TRACER(tcr)->traceopt(5, 0.5);
	TRACER(tcr)->traceopt(8, 1);//small block
	//TRACER(tcr)->traceopt(10, 1);
	//TRACER(tcr)->traceopt(11, 1);
	//TRACER(tcr)->traceopt(9, 4);
	//TRACER(tcr)->traceopt(4, 32);
	//TRACER(tcr)->traceopt(20, 0);
	TRACER(tcr)->traceopt(60, -1);
	//TRACER(tcr)->traceopt(5, 0.4);
	//TRACER(tcr)->traceopt(21, 1);
	TRACER(tcr)->traceopt(69, 777);
	//TRACER(tcr)->traceopt(17, 1);
	//TRACER(tcr)->traceopt(57, 2);//이 두개 조합은 효과 없음.
	//TRACER(tcr)->traceopt(59, 16);
	//TRACER(tcr)->traceopt(18, 2);
	//TRACER(tcr)->traceopt(64, 200);
	//TRACER(tcr)->traceopt(61, 1);
	//TRACER(tcr)->traceopt(62, 4);
	//TRACER(tcr)->traceopt(72, 512);
	//TRACER(tcr)->traceopt(73, sz / 10);
	//TRACER(tcr)->traceopt(3, 4);
	//TRACER(tcr)->traceopt(63, 0);
	TRACER(tcr)->traceopt(74, in_plus);

	TRACER(tcr)->traceopt(75, sz);
	TRACER(tcr)->traceopt(76, 1);
	//TRACER(tcr)->traceopt(77, 1);
	//TRACER(tcr)->traceopt(78, sz);
	//TRACER(tcr)->traceopt(79, 32);
	TRACER(tcr)->traceopt(0, 8);
	TRACER(tcr)->traceopt(63, 2);
	TRACER(tcr)->traceopt(62, -4);
	//in_gate = flux(tcr, { -1, x_seq_len, 1 }, tfloat, variable);
	//in_gate->arange(-1);
	//TRACER(tcr)->traceopt(101, 1);

	if(in_plus) tar_gate = flux(tcr, { -1, y_seq_len, 1 }, tfloat, variable);
	else tar_gate = flux(tcr, { -1, half_len + 1, 1 }, tfloat, variable);
	by_gate = flux(tcr, { -1, y_seq_len, 1 }, tfloat, variable);
	tcr->setbygate(by_gate, half_len + 1, az, az);//by gate에 입력과 타겟쌍을 구성할 경우
	//이 둘을 합한 discrete사이즈이고 입력파트가 없고 타겟만 있을 경우 타겟만의 discrete사이즈이다.
	//타겟만의 사이즈일 경우 바이게이트 전체는 추론을 위한 사이즈이고 이때 입력은 in_gate에 이전문맥으로 
	//구성한다.
	//by_gate->feedf(by);
	Generic cell(tcr, (Flux *)0, tar_gate, 235, batch_sz * half_len, batch_sz * half_len, 235);
	printf("mptest aaa\n");
	floatt sloss = 1, loss = 1;
	for(intt i = 0; loss > 0.007; i++) {
		//in_gate->feedf(ax);
		by_gate->feedf(by);
		tar_gate->feedf(ay);
		b = cell.train();
		//printf("\n");
		b->printo();
		printf("\n");
		loss = b->at_d(0);
		if(i == 0) sloss = loss;
		if((i + 1) % 5 == 0) {
			by_gate->fill(0.0);//reset
			by_gate->howrite(seq_d);//input
			by_gate->howrite(az, half_len);//padding
			c = cell.predict();
			c->printo(2, 3);
			printf("auto regression\n");
		}
	}
	printf("loss down: %f\n", sloss - b->at_d(0));
	/*
	//배치 1개씩 추론 테스트
	c = flux(tcr, { 1, y_seq_len, 1 }, tfloat, variable);
	c->fill(0.0);//reset
	seq_d = flux(tcr, { 1, half_len, 1 }, tfloat, variable);//input data
	seq_d->arange(-1);
	c->howrite(seq_d);//input
	by_gate->feedf(c);
	az->resizing4(1);
	*/
	//배치 여러개 추론 테스트
	//TRACER(tcr)->traceopt(55, 0);//auto encoding prediction test
	by_gate->fill(0.0);//reset
	by_gate->howrite(seq_d);//input
	by_gate->howrite(az, half_len);//padding
	
	c = cell.predict();
	c->printo(2,3);
	printf("auto regression\n");
	
	//tcr->saveWeight();
	//c = cell.predict(&loss);
	//c->printo();
	//loss->printo();
}
#include "cydrome/impulse.h"
void cnet_embedding_cross_entropy_dual_dgen1d_test(intt sz)
{
	Trace *tcr = (Trace *)trace(1);
	Flux *in_gate, *tar_gate, *by, *by_gate, *seq_d, *b = nullx, *ax, *ay, *az, *c;

	//tcr->modeset(1);

	intt x_seq_len = 64;
	intt y_seq_len = 64;
	intt batch_sz = 16;
	intt half_len = x_seq_len / 2 - 1;

	by = flux(tcr, { batch_sz, y_seq_len, 1 }, tfloat, variable);

	az = flux(tcr, { batch_sz, 1, 1 }, tfloat, variable);//token
	az->fill(0.0);

	seq_d = flux(tcr, { batch_sz, half_len, 1 }, tfloat, variable);//input data
	seq_d->arange(-1);

	by->howrite(seq_d);//input
	//by->printo(2);
	by->howrite(az, half_len);//padding
	//by->printo(2);
	by->howrite(az, half_len + 1);//go token
	by->howrite(seq_d, half_len + 2);//target

	by->printo(2, 3);

	ax = flux(tcr, { batch_sz, x_seq_len, 1 }, tfloat, variable);//prev input data
	ax->arange(-1);
	//ax->printo(1, 0);

	ay = flux(tcr, { batch_sz, half_len + 1, 1 }, tfloat, variable);
	ay->howrite(seq_d);//target
	ay->howrite(az, half_len);//end token
	ay->printo(2, 5);

	//tcr->sizeBatch(4);
	TRACER(tcr)->traceopt(1, 1);
	TRACER(tcr)->traceopt(0, 4);//8
	//TRACER(tcr)->ebatch_t = 1;
	//TRACER(tcr)->strideJump = 3;
	//TRACER(tcr)->traceopt(6, 1);

	//TRACER(tcr)->traceopt(5, 0.5);
	TRACER(tcr)->traceopt(8, 1);//small block
	//TRACER(tcr)->traceopt(11, 1);
	//TRACER(tcr)->traceopt(9, 4);
	//TRACER(tcr)->traceopt(4, 32);
	//TRACER(tcr)->traceopt(20, 0);
	//TRACER(tcr)->traceopt(57, 2);//이 두개 조합은 효과 없음.
	//TRACER(tcr)->traceopt(59, 16);
	TRACER(tcr)->traceopt(53, 1);//1d
	//TRACER(tcr)->traceopt(9, 0);
	//TRACER(tcr)->traceopt(10, 0);
	//TRACER(tcr)->traceopt(60, 1);
	//TRACER(tcr)->traceopt(61, 1);
	//TRACER(tcr)->traceopt(62, 2);
	//TRACER(tcr)->traceopt(63, 1);
	//TRACER(tcr)->traceopt(64, 4);
	TRACER(tcr)->traceopt(65, 0);
	TRACER(tcr)->traceopt(54, 5.0);//cross entropy 는 오차 값이 크므로 
	TRACER(tcr)->traceopt(69, 777);
	printf("batch size: %d\n", batch_sz);
	intt hidden_sz = 32, embed_sz = 16, in_discret = batch_sz * x_seq_len;
	in_gate = flux(tcr, { -1, x_seq_len, 1 }, tfloat, variable);
	tar_gate = flux(tcr, { -1, half_len + 1, 1 }, tfloat, variable);
	by_gate = flux(tcr, { -1, y_seq_len, 1 }, tfloat, variable);
	//by gate에 입력과 타겟쌍을 구성할 경우 이 둘을 합한 discrete사이즈이고 입력파트가 없고 타겟만 있을
	//경우 타겟만의 discrete사이즈이다. 타겟만의 사이즈일 경우 바이게이트 전체는 추론을 위한 사이즈이고 
	//이때 입력은 in_gate에 이전문맥으로 구성한다. 다이나젠은 이전 문맥또는 입력 파트가 큰경우 이를 
	//압축하기위해 사용한다. embedim은 듀얼 인코더의 임베딩 사이즈로 반드시 설정하고 latent_sz, 
	//indiscret는 듀얼의 사이즐 다이나젠 생성자에서 명시한 것과 다르게 할 경우 설정한다. outdiscret는
	//다이나젠 생성자에서 듀얼 인코더의 것이 명시되므로 따로 설정 필요없다.
	tcr->setbygate(by_gate, half_len + 1, az, az, embed_sz, hidden_sz, in_discret);
	
	Impulse im(tcr);
	Dynagen dgen(T_DG_CALSS, &im, in_gate, tar_gate, hidden_sz, in_discret, batch_sz * y_seq_len, embed_sz);
	by_gate = dgen.getbygate();
	Flux *loss;
	intt rv = 1;
	for(intt i = 1; i < 100000; i++) {
		//tcr->printWeight();
		if(rv) {
			in_gate->feedf(ax);
			tar_gate->feedf(ay);
			by_gate->feedf(by);
			loss = im.train(&rv);
		}
		//loss->printo();
		//dgen.dgenloss[0]->printo();
		if(rv == 0) {
			//if(i % 10 == 0) {
			printf("----------- %d ---------\n", i);
			by->fill(0.0); //reset #원래는 전체 리셋하지 않고 밑에서 go token만 적재해야 하나 확실확인하기위해
			by->howrite(seq_d);//input
			by->howrite(az, half_len);//padding
			by->howrite(az, half_len + 1);//go token
			by_gate->feedf(by);
			in_gate->feedf(ax);
			tar_gate->feedf(ay);
			c = im.predict(&loss);
			//c = dgen.predictv(&loss);
			loss->printo();
			c->printo(2, 10);
			break;
		}
	}
	/*
	//Generic cell(tcr, in_gate, tar_gate, hidden_sz, batch_sz * half_len, batch_sz * half_len, 16, sz);
	floatt sloss = 1, loss = 1;
	for(intt i = 0; loss > 0.007; i++) {
		in_gate->feedf(ax);
		by_gate->feedf(by);
		tar_gate->feedf(ay);
		b = cell.train();
		//printf("\n");
		b->printo();
		printf("\n");
		loss = b->at_d(0);
		if(i == 0) sloss = loss;
	}
	printf("loss down: %f\n", sloss - b->at_d(0));

	//배치 1개씩 추론 테스트
	//c = flux(tcr, { 1, y_seq_len, 1 }, tfloat, variable);
	//c->fill(0.0);//reset
	//seq_d = flux(tcr, { 1, half_len, 1 }, tfloat, variable);//input data
	//seq_d->arange(-1);
	//c->howrite(seq_d);//input
	//by_gate->feedf(c);
	//az->resizing4(1);

	//배치 여러개 추론 테스트
	//TRACER(tcr)->traceopt(55, 0);//auto encoding prediction test
	by_gate->fill(0.0);//reset
	by_gate->howrite(seq_d);//input
	by_gate->howrite(az, half_len);//padding

	c = cell.predict();
	c->printo(2, 3);
	printf("auto regression\n");

	//tcr->saveWeight();
	//c = cell.predict(&loss);
	//c->printo();
	//loss->printo();
	*/
}
void cnet_mse_dual_dgen1d_64y__test(void)
{
	Trace *tcr = (Trace *)trace(1);
	Flux *in_gate, *tar_gate, *by, *by_gate, *seq_d, *b = nullx, *ax, *ay, *az, *c;

	//tcr->modeset(1);

	intt x_seq_len = 64;
	intt y_seq_len = 64;
	intt batch_sz = 16;
	intt half_len = x_seq_len / 2 - 1;

	by = flux(tcr, { batch_sz, y_seq_len, 1 }, tfloat, variable);

	az = flux(tcr, { batch_sz, 1, 1 }, tfloat, variable);//token
	az->fill(0.0);

	seq_d = flux(tcr, { batch_sz, half_len, 1 }, tfloat, variable);//input data
	seq_d->arange(-1);
	seq_d->stdnormal();

	by->howrite(seq_d);//input
	//by->printo(2);
	by->howrite(az, half_len);//padding
	//by->printo(2);
	by->howrite(az, half_len + 1);//go token
	by->howrite(seq_d, half_len + 2);//target

	by->printo(2, 3);

	ax = flux(tcr, { batch_sz, x_seq_len, 1 }, tfloat, variable);//prev input data
	ax->arange(-1);
	//ax->printo(1, 0);
	ax->stdnormal();

	ay = flux(tcr, { batch_sz, half_len + 1, 1 }, tfloat, variable);
	ay->howrite(seq_d);//target
	ay->howrite(az, half_len);//end token
	ay->printo(2, 5);

	//tcr->sizeBatch(4);
	TRACER(tcr)->traceopt(1, 1);
	TRACER(tcr)->traceopt(0, 4);//8
	//TRACER(tcr)->ebatch_t = 1;
	//TRACER(tcr)->strideJump = 3;
	//TRACER(tcr)->traceopt(6, 1);

	//TRACER(tcr)->traceopt(5, 0.5);
	TRACER(tcr)->traceopt(8, 1);//small block
	//TRACER(tcr)->traceopt(11, 1);
	//TRACER(tcr)->traceopt(9, 4);
	//TRACER(tcr)->traceopt(4, 32);
	//TRACER(tcr)->traceopt(20, 0);
	//TRACER(tcr)->traceopt(57, 2);//이 두개 조합은 효과 없음.
	//TRACER(tcr)->traceopt(59, 16);
	TRACER(tcr)->traceopt(53, 1);//1d
	//TRACER(tcr)->traceopt(9, 0);
	//TRACER(tcr)->traceopt(10, 0);
	//TRACER(tcr)->traceopt(60, 1);
	//TRACER(tcr)->traceopt(61, 1);
	//TRACER(tcr)->traceopt(62, 2);
	//TRACER(tcr)->traceopt(63, 1);
	//TRACER(tcr)->traceopt(64, 4);
	TRACER(tcr)->traceopt(65, 0);
	TRACER(tcr)->traceopt(54, 0.07);//cross entropy 는 오차 값이 크므로 
	TRACER(tcr)->traceopt(69, 777);
	printf("batch size: %d\n", batch_sz);
	intt hidden_sz = 32;
	in_gate = flux(tcr, { -1, x_seq_len, 1 }, tfloat, variable);
	tar_gate = flux(tcr, { -1, half_len + 1, 1 }, tfloat, variable);
	by_gate = flux(tcr, { -1, y_seq_len, 1 }, tfloat, variable);
	//by gate에 입력과 타겟쌍을 구성할 경우 이 둘을 합한 discrete사이즈이고 입력파트가 없고 타겟만 있을
	//경우 타겟만의 discrete사이즈이다. 타겟만의 사이즈일 경우 바이게이트 전체는 추론을 위한 사이즈이고 
	//이때 입력은 in_gate에 이전문맥으로 구성한다. 다이나젠은 이전 문맥또는 입력 파트가 큰경우 이를 
	//압축하기위해 사용한다. embedim은 듀얼 인코더의 임베딩 사이즈로 반드시 설정하고 latent_sz, 
	//indiscret는 듀얼의 사이즐 다이나젠 생성자에서 명시한 것과 다르게 할 경우 설정한다. outdiscret는
	//다이나젠 생성자에서 듀얼 인코더의 것이 명시되므로 따로 설정 필요없다.
	tcr->setbygate(by_gate, half_len + 1, az, az, 0, hidden_sz, 0);

	Impulse im(tcr);
	Dynagen dgen(T_DG_CALSS, &im, in_gate, tar_gate, hidden_sz, 0, 0, 0);
	by_gate = dgen.getbygate();
	Flux *loss;
	intt rv = 1;
	for(intt i = 1; i < 100000; i++) {
		//tcr->printWeight();
		if(rv) {
			in_gate->feedf(ax);
			tar_gate->feedf(ay);
			by_gate->feedf(by);
			loss = im.train(&rv);
		}
		//loss->printo();
		//dgen.dgenloss[0]->printo();
		if(rv == 0) {
			//if(i % 10 == 0) {
			printf("----------- %d ---------\n", i);
			by->fill(0.0); //reset #원래는 전체 리셋하지 않고 밑에서 go token만 적재해야 하나 확실확인하기위해
			by->howrite(seq_d);//input
			by->howrite(az, half_len);//padding
			by->howrite(az, half_len + 1);//go token
			by_gate->feedf(by);
			in_gate->feedf(ax);
			tar_gate->feedf(ay);
			c = im.predict(&loss);
			//c = dgen.predictv(&loss);
			loss->printo();
			c->printo(2, 10);
			break;
		}
	}
}
void cnet_mse_dual_dgen1d_16y_test(void)
{
	Trace *tcr = (Trace *)trace(1);
	Flux *in_gate, *tar_gate, *by, *by_gate, *seq_d, *b = nullx, *ax, *ay, *az, *c;

	//tcr->modeset(1);
	tcr->npset(1000);
	intt x_prev_len = 256;
	intt y_seq_len = 16, x_seq_len = 0;
	intt batch_sz = 16;
	intt data_len = y_seq_len - 1;

	by = flux(tcr, { batch_sz, x_seq_len + y_seq_len, 1 }, tfloat, variable);

	az = flux(tcr, { batch_sz, 1, 1 }, tfloat, variable);//token
	az->fill(0.0);

	seq_d = flux(tcr, { batch_sz, data_len, 1 }, tfloat, variable);//input data
	seq_d->arange(-1);
	seq_d->stdnormal();

	by->howrite(az, x_seq_len);//go token
	by->howrite(seq_d, x_seq_len + 1);//target

	by->printo(2, 3);

	ax = flux(tcr, { batch_sz, x_prev_len, 1 }, tfloat, variable);//prev input data
	ax->arange(-1);
	//ax->printo(1, 0);
	ax->stdnormal();

	ay = flux(tcr, { batch_sz, y_seq_len, 1 }, tfloat, variable);
	ay->howrite(seq_d);//target
	ay->howrite(az, data_len);//end token
	ay->printo(2, 5);

	//tcr->sizeBatch(4);
	TRACER(tcr)->traceopt(1, 1);
	TRACER(tcr)->traceopt(0, 4);//8
	//TRACER(tcr)->ebatch_t = 1;
	//TRACER(tcr)->strideJump = 3;
	//TRACER(tcr)->traceopt(6, 1);

	//TRACER(tcr)->traceopt(5, 0.5);
	TRACER(tcr)->traceopt(8, 1);//small block
	//TRACER(tcr)->traceopt(11, 1);
	//TRACER(tcr)->traceopt(9, 4);
	//TRACER(tcr)->traceopt(4, 32);
	//TRACER(tcr)->traceopt(20, 0);
	//TRACER(tcr)->traceopt(57, 2);//이 두개 조합은 효과 없음.
	//TRACER(tcr)->traceopt(59, 16);
	TRACER(tcr)->traceopt(53, 1);//1d
	//TRACER(tcr)->traceopt(9, 0);
	//TRACER(tcr)->traceopt(10, 0);
	TRACER(tcr)->traceopt(60, -1);
	//TRACER(tcr)->traceopt(61, 1);
	//TRACER(tcr)->traceopt(62, 2);
	TRACER(tcr)->traceopt(63, 0);
	//TRACER(tcr)->traceopt(64, 4);
	TRACER(tcr)->traceopt(65, 0);
	TRACER(tcr)->traceopt(54, 0.1);//cross entropy 는 오차 값이 크므로 
	TRACER(tcr)->traceopt(69, 777);
	TRACER(tcr)->traceopt(43, 32);
	//TRACER(tcr)->traceopt(71, 4);
	printf("batch size: %d\n", batch_sz);
	intt hidden_sz = 32;
	in_gate = flux(tcr, { -1, x_prev_len, 1 }, tfloat, variable);
	tar_gate = flux(tcr, { -1, y_seq_len, 1 }, tfloat, variable);
	by_gate = flux(tcr, { -1, x_seq_len + y_seq_len, 1 }, tfloat, variable);
	//by gate에 입력과 타겟쌍을 구성할 경우 이 둘을 합한 discrete사이즈이고 입력파트가 없고 타겟만 있을
	//경우 타겟만의 discrete사이즈이다. 타겟만의 사이즈일 경우 바이게이트 전체는 추론을 위한 사이즈이고 
	//이때 입력은 in_gate에 이전문맥으로 구성한다. 다이나젠은 이전 문맥또는 입력 파트가 큰경우 이를 
	//압축하기위해 사용한다. embedim은 듀얼 인코더의 임베딩 사이즈로 반드시 설정하고 latent_sz, 
	//indiscret는 듀얼의 사이즐 다이나젠 생성자에서 명시한 것과 다르게 할 경우 설정한다. outdiscret는
	//다이나젠 생성자에서 듀얼 인코더의 것이 명시되므로 따로 설정 필요없다.
	tcr->setbygate(by_gate, y_seq_len, az, az, 0);

	Impulse im(tcr);
	Dynagen dgen(T_DG_CALSS, &im, in_gate, tar_gate, hidden_sz, 0, 0, 0);
	by_gate = dgen.getbygate();
	Flux *loss;
	intt rv = 1;
	for(intt i = 1; i < 100000; i++) {
		//tcr->printWeight();
		if(rv) {
			by_gate->feedf(by);
			in_gate->feedf(ax);
			tar_gate->feedf(ay);
			loss = im.train(&rv);
		}
		//loss->printo();
		//dgen.dgenloss[0]->printo();
		if((i + 1) % 10 == 0 || rv == 0) {
			//if(i % 10 == 0) {
			printf("----------- %d ---------\n", i);
			by->fill(0.0); //reset #원래는 전체 리셋하지 않고 밑에서 go token만 적재해야 하나 확실확인하기위해
			by->howrite(az, x_seq_len);//go token
			by_gate->feedf(by);
			in_gate->feedf(ax);
			tar_gate->feedf(ay);
			c = im.predict(&loss);
			//c = dgen.predictv(&loss);
			loss->printo();
			c->printo(2, 10);
			if(rv == 0) break;
		}
	}
}
void algol_embedding_cross_entropy_convolution_test(intt sz)
{
	Tracer *tcr = trace(0);
	Flux *a, *a2, *b, *c, *w, *t, *aa, *ax, *ay;

	//tcr->modeset(1);
	//tcr->npset(17);

	intt x_seq_len = 32;//;//56;//32;
	intt y_seq_len = 32;//7;//32;
	intt batch_sz = 4;
	ax = flux(tcr, { batch_sz, x_seq_len, 1 }, tfloat, variable);
	ax->arange(-1);
	//ax->randn(0, 1);
	ay = flux(tcr, { batch_sz, y_seq_len, 1 }, tfloat, variable);
	ay->arange(-1);

	TRACER(tcr)->traceopt(1, 1);
	TRACER(tcr)->traceopt(0, 4);
	TRACER(tcr)->traceopt(8, 1);//small block
	//TRACER(tcr)->traceopt(11, 1);
	TRACER(tcr)->traceopt(10, 1);
	TRACER(tcr)->traceopt(9, 4);
	//TRACER(tcr)->traceopt(20, 2);
	//TRACER(tcr)->traceopt(24, 1);
	//TRACER(tcr)->traceopt(28, -2);

	a = flux(tcr, { -1, x_seq_len, 1 }, tfloat, variable);
	a->arange(-1);
	a2 = flux(tcr, { -1, y_seq_len, 1 }, tfloat, variable);
	//a->arange(-1);
	//Algol cell(a, a2, 32, batch_sz * x_seq_len, batch_sz * x_seq_len, 8, 1, ACTF_RELU);
	Algol cell(a, a2, 32, 0, 0, 0, 1, ACTF2_PRELU);
	//Algol cell(a, a2, 32, 0, 0, 0, 0);

	Flux *pred_loss;
	for(intt i = 1;i < 100000; i++) {

		a->feedf(ax);
		a2->feedf(ay);
		b = cell.train();
		//printf("gen loss--\n");
		//b->printo();
		//printf("dsc loss\n");
		//cell.agloss2->printo();
		//printf("\n");
		//printf("\n");
		//printf("\n");
		if(sz > 0 && i % 10 == 0) {
			printf((bytet *)"--++--\n");
			a->feedf(ax);
			c = cell.predict(&pred_loss);
			//c->printo();
			printf("pred gen loss\n");
			pred_loss->printo();
			//if(sz > 1) tcr->printWeight();
		}
	}
}
void stratus_embedding_cross_entropy_convolution_test(intt sz)
{
	Tracer *tcr = trace(0);
	Flux *a, *a2, *b, *c, *w, *t, *aa, *ax, *ay, *ax_test, *ay_test;

	//tcr->modeset(1);

	intt x_seq_len = 64;//56;//32;
	intt y_seq_len = 64;//7;//32;
	intt batch_sz = 16;
	ax = flux(tcr, { batch_sz, x_seq_len, 1 }, tfloat, variable);
	ax->arange(-1);
	ay = flux(tcr, { batch_sz, y_seq_len, 1 }, tfloat, variable);
	ay->arange(-1);

	//ax_test = flux(tcr, { 1, x_seq_len, 1 }, tfloat, variable);
	//ax_test->arange(-1);
	//ay_test = flux(tcr, { 1, y_seq_len, 1 }, tfloat, variable);
	//ay_test->arange(-1);
	TRACER(tcr)->traceopt(1, 1);
	TRACER(tcr)->traceopt(0, 4);
	TRACER(tcr)->traceopt(8, 1);//small block
	TRACER(tcr)->traceopt(11, 1);
	TRACER(tcr)->traceopt(6, 1);//1아면 인코더 네트워크에서만 splot lat적용, 2이면 인코더-디코더 네크워크에서 splot lat적용
	TRACER(tcr)->traceopt(9, 4);
	TRACER(tcr)->traceopt(16, 3);
	TRACER(tcr)->traceopt(21, 1);//어텐션은 위 6에 따라 젹용 범위 결정.
	TRACER(tcr)->traceopt(26, -1);
	TRACER(tcr)->traceopt(34, 1);
	/*
	TRACER(tcr)->traceopt(1, 1);
	TRACER(tcr)->traceopt(0, 4);
	TRACER(tcr)->traceopt(8, 1);//small block
	//TRACER(tcr)->traceopt(13, 2);
	TRACER(tcr)->traceopt(10, 1);
	//TRACER(tcr)->traceopt(11, 1);
	TRACER(tcr)->traceopt(6, 2);//1아면 인코더 네트워크에서만 splot lat적용, 2이면 인코더-디코더 네크워크에서 splot lat적용
	TRACER(tcr)->traceopt(9, 4);
	//TRACER(tcr)->traceopt(15, 1);//layer normal
	//TRACER(tcr)->traceopt(24, 1);
	//TRACER(tcr)->traceopt(20, 3);
	//TRACER(tcr)->traceopt(18, 2);
	//TRACER(tcr)->traceopt(16, 3);
	//TRACER(tcr)->traceopt(21, 1);//어텐션은 위 6에 따라 젹용 범위 결정.
	TRACER(tcr)->traceopt(26, 2);
	TRACER(tcr)->traceopt(27, 1);
	//TRACER(tcr)->traceopt(3, 6);
	//TRACER(tcr)->traceopt(29, 1);
	//TRACER(tcr)->traceopt(30, 1);
	//TRACER(tcr)->traceopt(31, 2);
	//TRACER(tcr)->traceopt(7, 1);
	TRACER(tcr)->traceopt(32, -2);
	TRACER(tcr)->traceopt(33, 6.80);
	//TRACER(tcr)->traceopt(34, 1);
	*/
	a = flux(tcr, { -1, x_seq_len, 1 }, tfloat, variable);
	a->arange(-1);
	a2 = flux(tcr, { -1, y_seq_len, 1 }, tfloat, variable);
	//a->arange(-1);
	Stratus cell(a, a2, 32, batch_sz * x_seq_len, batch_sz * x_seq_len, 8);

	Flux *pred_loss;
	for(intt i = 0;i < 100; i++) {

		a->feedf(ax);
		a2->feedf(ay);
		b = cell.train();
		printf("sor loss\n");
		b->printo();
		printf("tar loss\n");
		cell.srtar_loss->printo();
		printf("\n");
		printf("\n");
		printf("\n");
		if(sz > 0 && i % 1 == 0) {
			printf((bytet *)"----++--\n");
			//a->feedf(ax_test);
			//a2->feedf(ay_test);
			c = cell.predict(&pred_loss);
			//c->printo();
			printf("pred gen loss\n");
			pred_loss->printo();
			if(sz > 1) tcr->printWeight();
			//tcr->saveWeight();
		}
	}
	//tcr->saveWeight();
}
void stratus_embedding_cross_entropy_convolution_test2(intt sz)
{
	Tracer *tcr = trace(0);
	Flux *a, *a2, *b, *c, *w, *t, *aa, *ax, *ay, *ax_test, *ay_test;
	Flux *x[10], *y[10];
	//tcr->modeset(1);

	intt x_seq_len = 64;//56;//32;
	intt y_seq_len = 64;//7;//32;
	intt batch_sz = 16;

	for(intt i = 0;i < 10; i++) {
		ax = flux(tcr, { batch_sz, x_seq_len, 1 }, tfloat, variable);
		ax->randn(0, 1);
		ay = flux(tcr, { batch_sz, y_seq_len, 1 }, tfloat, variable);
		ay->randn(0, 1);
		x[i] = ax;
		y[i] = ay;
	}
	//ax_test = flux(tcr, { 1, x_seq_len, 1 }, tfloat, variable);
	//ax_test->arange(-1);
	//ay_test = flux(tcr, { 1, y_seq_len, 1 }, tfloat, variable);
	//ay_test->arange(-1);


	TRACER(tcr)->traceopt(1, 1);
	TRACER(tcr)->traceopt(0, 4);
	TRACER(tcr)->traceopt(8, 1);//small block
	//TRACER(tcr)->traceopt(13, 2);
	TRACER(tcr)->traceopt(10, 1);
	//TRACER(tcr)->traceopt(11, 1);
	TRACER(tcr)->traceopt(6, 1);//1아면 인코딩에서만 splot lat적용, 2이면 인코딩/디코딩에서 splot lat적용
	TRACER(tcr)->traceopt(9, 4);
	//TRACER(tcr)->traceopt(15, 1);//layer normal
	//TRACER(tcr)->traceopt(24, 1);
	TRACER(tcr)->traceopt(20, 1);
	//TRACER(tcr)->traceopt(18, 2);
	TRACER(tcr)->traceopt(16, 3);
	TRACER(tcr)->traceopt(21, 1);//어텐션은 위 6에 따라 젹용 범위 결정.
	TRACER(tcr)->traceopt(26, 3);
	TRACER(tcr)->traceopt(27, 1);
	//TRACER(tcr)->traceopt(3, 6);
	//TRACER(tcr)->traceopt(29, 1);
	//TRACER(tcr)->traceopt(30, 1);
	//TRACER(tcr)->traceopt(31, 2);
	//TRACER(tcr)->traceopt(7, 1);
	TRACER(tcr)->traceopt(32, -10);
	TRACER(tcr)->traceopt(33, 10.0);

	a = flux(tcr, { -1, x_seq_len, 1 }, tfloat, variable);
	a->arange(-1);
	a2 = flux(tcr, { -1, y_seq_len, 1 }, tfloat, variable);
	//a->arange(-1);
	Stratus cell(a, a2, 32, batch_sz * x_seq_len, batch_sz * x_seq_len, 8);

	Flux *pred_loss;
	for(intt i = 0;i < 100; i++) {

		a->feedf(x[i % 10]);
		a2->feedf(y[i % 10]);
		b = cell.train();
		printf("sor loss\n");
		b->printo();
		printf("tar loss\n");
		cell.srtar_loss->printo();
		printf("\n");
		printf("\n");
		printf("\n");
		/*if(sz > 0 && i % 1 == 0) {
			printf((bytet *)"----++--\n");
			//a->feedf(ax_test);
			//a2->feedf(ay_test);
			c = cell.predict(&pred_loss);
			//c->printo();
			printf("pred gen loss\n");
			pred_loss->printo();
			if(sz > 1) tcr->printWeight();
			//tcr->saveWeight();
		}*/
	}
	//tcr->saveWeight();
}
void dual_image2seq_test(intt gid)
{
	Trace *tcr = (Trace *)trace(1);
	Flux *prev_gate, *y_gate, *by_feed, *by_gate, *xseq_v, *yseq_v, *b = nullx, *prev_feed, *y_feed, *az, *c;

	//tcr->modeset(1);
	tcr->npset(1000);
	tcr->setgpudev(gid);
	printf("SET GPU DEV: %d\n", gid);
	intt y_regress_len = 8, x_seq_len = 8, prev_batch = 16;
	intt batch_sz = prev_batch / x_seq_len;//듀얼인코더의 배치 산출, prev의 배치 한개는 듀얼인코더의 한개 시퀀스에 대응
	intt yseq_len = y_regress_len - 1;

	by_feed = flux(tcr, { batch_sz, x_seq_len + y_regress_len, 1 }, tfloat, variable);

	az = flux(tcr, { batch_sz, 1, 1 }, tfloat, variable);//token
	az->fill(0.0);

	xseq_v = flux(tcr, { batch_sz, x_seq_len, 1 }, tfloat, variable);//input data
	xseq_v->arange(-1);
	xseq_v->stdnormal();
	yseq_v = flux(tcr, { batch_sz, yseq_len, 1 }, tfloat, variable);//target data
	yseq_v->arange(-1);
	yseq_v->stdnormal();

	by_feed->howrite(xseq_v, 0);//
	by_feed->howrite(az, x_seq_len);//go token
	by_feed->howrite(yseq_v, x_seq_len + 1);//target

	by_feed->printo(2, 3);

	prev_feed = flux(tcr, { prev_batch, 84, 84, 1 }, tfloat, variable);//prev input data
	prev_feed->arange(-1);
	//prev_feed->printo(1, 0);
	prev_feed->stdnormal();

	y_feed = flux(tcr, { batch_sz, y_regress_len, 1 }, tfloat, variable);
	y_feed->howrite(yseq_v);//target
	y_feed->howrite(az, yseq_len);//end token
	y_feed->printo(2, 5);

	//tcr->sizeBatch(4);
	TRACER(tcr)->traceopt(1, 1);
	TRACER(tcr)->traceopt(0, 4);//8
	//TRACER(tcr)->ebatch_t = 1;
	//TRACER(tcr)->strideJump = 3;
	//TRACER(tcr)->traceopt(6, 1);

	//TRACER(tcr)->traceopt(5, 0.5);
	TRACER(tcr)->traceopt(8, 1);//small block
	//TRACER(tcr)->traceopt(11, 1);
	//TRACER(tcr)->traceopt(9, 4);
	//TRACER(tcr)->traceopt(4, 32);
	//TRACER(tcr)->traceopt(20, 0);
	//TRACER(tcr)->traceopt(57, 2);//이 두개 조합은 효과 없음.
	//TRACER(tcr)->traceopt(59, 16);
	TRACER(tcr)->traceopt(53, 2);//1d
	TRACER(tcr)->traceopt(54, 0.2);//cross entropy 는 오차 값이 크므로 
	TRACER(tcr)->traceopt(69, 777);
	TRACER(tcr)->traceopt(75, 2);
	TRACER(tcr)->traceopt(86, 1);
	TRACER(tcr)->traceopt(88, -1);
	TRACER(tcr)->traceopt(90, x_seq_len);
	TRACER(tcr)->traceopt(91, 1);
	TRACER(tcr)->traceopt(43, 512);
	TRACER(tcr)->traceopt(92, 1);
	printf("batch size 777777777777 %d\n", batch_sz);
	intt hidden_sz = 8;
	prev_gate = flux(tcr, { -1, 84, 84, 1 }, tfloat, variable);
	y_gate = flux(tcr, { -1, y_regress_len, 1 }, tfloat, variable);
	by_gate = flux(tcr, { -1, x_seq_len + y_regress_len, 1 }, tfloat, variable);
	//by_feed gate에 입력과 타겟쌍을 구성할 경우 이 둘을 합한 discrete사이즈이고 입력파트가 없고 타겟만 있을
	//경우 타겟만의 discrete사이즈이다. 타겟만의 사이즈일 경우 바이게이트 전체는 추론을 위한 사이즈이고 
	//이때 입력은 prev_gate에 이전문맥으로 구성한다. 다이나젠은 이전 문맥또는 입력 파트가 큰경우 이를 
	//압축하기위해 사용한다. embedim은 듀얼 인코더의 임베딩 사이즈로 반드시 설정하고 latent_sz, 
	//indiscret는 듀얼의 사이즐 다이나젠 생성자에서 명시한 것과 다르게 할 경우 설정한다. outdiscret는
	//다이나젠 생성자에서 듀얼 인코더의 것이 명시되므로 따로 설정 필요없다.
	tcr->setbygate(by_gate, y_regress_len, az, az, 0);

	Impulse im(tcr);
	Dynagen dgen(T_DG_CALSS, &im, prev_gate, y_gate, hidden_sz, 0, 0, 0);
	by_gate = dgen.getbygate();
	Flux *loss;
	intt rv = 1;
	for(intt i = 1; i < 100000; i++) {
		//tcr->printWeight();
		if(rv) {
			by_gate->feedf(by_feed);
			prev_gate->feedf(prev_feed);
			y_gate->feedf(y_feed);
			loss = im.train(&rv);
		}
		//loss->printo();
		//dgen.dgenloss[0]->printo();
		if((i + 1) % 10 == 0 || rv == 0) {
			//if(i % 10 == 0) {
			//printf("----------- %d ---------\n", i);
			by_feed->fill(0.0); //reset #원래는 전체 리셋하지 않고 밑에서 go token만 적재해야 하나 확실확인하기위해
			by_feed->howrite(xseq_v, 0);
			by_feed->howrite(az, x_seq_len);//go token
			by_gate->feedf(by_feed);
			prev_gate->feedf(prev_feed);
			y_gate->feedf(y_feed);
			c = im.predict(&loss);
			//c = dgen.predictv(&loss);
			loss->printo();
			//c->printo(2, 10);
			if(rv == 0) break;
		}
	}
}
void dynagen_test2d_mse()
{
	Tracer *tcr = trace(1);
	Flux *a, *a2, *b, *c, *w, *t, *aa, *ax, *ay;
	intt x_seq_len = 16;//1//56;//32;
	intt y_seq_len = 16;//1;//7;//32;
	intt batch_sz = 5;
	intx feat_sz = 1;
	intt hidden_sz = 32;
	//doublet mincode, maxcode;
	printf("2dmse bbb\n");

	//tcr->modeset(1);
	ax = flux(tcr, { batch_sz, x_seq_len, x_seq_len, feat_sz }, tfloat, variable);
	ax->arange(-1);
	//aa->randn(0, 1);
	//aa->minmax(mincode, maxcode);
	//ax = *(*aa - mincode) / (floatt)(maxcode - mincode);
	ax->stdnormal();
	ax->printo(2, 10);
	
	printf("\n");
	printf("\n");
	printf("\n");

	//aa->minmax(mincode, maxcode);
	//aa->minmax_normal(mincode, maxcode);
	//aa->printo(2, 10);

	//ay = flux(tcr, { batch_sz, y_seq_len, y_seq_len, feat_sz }, tfloat, variable);
	//ay->arange(-1);
	//ay->randn(0, 1);
	//tcr->sizeBatch(4);
	TRACER(tcr)->traceopt(1, 1);
	//TRACER(tcr)->ebatch_t = 1;
	//TRACER(tcr)->strideJump = 3;
	//TRACER(tcr)->traceopt(6, 1);
	//TRACER(tcr)->traceopt(9, 2);
	//TRACER(tcr)->traceopt(5, 0.5);
	TRACER(tcr)->traceopt(8, 1);//small block
	TRACER(tcr)->traceopt(10, 1);// 5);//reduce dot
	//TRACER(tcr)->traceopt(11, 1);
	TRACER(tcr)->traceopt(9, 4);
	//TRACER(tcr)->traceopt(16, 2);
	//TRACER(tcr)->traceopt(14, 40);
	//TRACER(tcr)->traceopt(15, 2);//layer normal
	//TRACER(tcr)->traceopt(21, 1);//어텐션 수행
	//TRACER(tcr)->traceopt(16, 3);//최종 인코딩에서만 어텐션 수행
	//TRACER(tcr)->traceopt(12, 64);
	//TRACER(tcr)->traceopt(17, 1);
	//TRACER(tcr)->traceopt(18, 100);//prelu
	//TRACER(tcr)->traceopt(20, 3);//by pass print
	//TRACER(tcr)->traceopt(22, 1);//optimizier
	//TRACER(tcr)->traceopt(23, 0.1);//learnning rate
	//TRACER(tcr)->traceopt(7, 1);
	//TRACER(tcr)->traceopt(51, 1);

	TRACER(tcr)->traceopt(53, 2);//1d
	TRACER(tcr)->traceopt(42, 1);
	//TRACER(tcr)->traceopt(52, 1);//동시학습
	TRACER(tcr)->traceopt(43, 16);
	TRACER(tcr)->traceopt(44, 16);
	//TRACER(tcr)->traceopt(47, 0);//overlap slide
	//TRACER(tcr)->traceopt(54, 0.1);
	a = flux(tcr, { -1, x_seq_len, x_seq_len, feat_sz }, tfloat, variable);
	//a->arange(-1);
	a2 = flux(tcr, { -1, y_seq_len, y_seq_len, feat_sz }, tfloat, variable);

	Impulse im(tcr);
	im.trainStep(3000);
	Dynagen dgen(T_DG_TRANSLATE, &im, a, a2, hidden_sz, 0, 0, 0, ACTF2_PRELU);
	Flux *loss;
	intt rv = 1;
	for(intt i = 1; i < 100000; i++) {
		//tcr->printWeight();
		if(rv) {
			a->feedf(ax);
			a2->feedf(ax);
			loss = im.train(&rv);
		}
		//loss->printo();
		//dgen.dgenloss[0]->printo();
		if(rv == 0) {
			//if(i % 3 == 0) {
			printf("----------- %d ---------\n", i);
			printf("\n");
			printf("\n");
			printf("\n");
			printf("--------------------\n");
			a->feedf(ax);
			a2->feedf(ax);
			c = im.predict(&loss);
			//c = dgen.predictv(&loss);
			loss->printo();
			c->printo(2, 10);
		}
	}
	printf("ddd\n");
}
void dynagen_test2d()
{
	Tracer *tcr = trace(0);
	Flux *a, *a2, *b, *c, *w, *t, *aa, *ax, *ay;
	intt x_seq_len = 16;//1//56;//32;
	intt y_seq_len = 16;//1;//7;//32;
	intt batch_sz = 5;
	intx feat_sz = 1;//discret인 경우 1로 해야 함.
	intt hidden_sz = 32;
	printf("2d bbb\n");

	//tcr->modeset(1);
	ax = flux(tcr, { batch_sz, x_seq_len, x_seq_len, feat_sz }, tfloat, variable);
	ax->arange(-1);
	//ax->randn(0, 1);
	ay = flux(tcr, { batch_sz, y_seq_len, y_seq_len, feat_sz }, tfloat, variable);
	ay->arange(-1);
	//ay->randn(0, 1);
	ax->printo(2, 10);
	//tcr->sizeBatch(4);
	TRACER(tcr)->traceopt(1, 1);
	//TRACER(tcr)->ebatch_t = 1;
	//TRACER(tcr)->strideJump = 3;
	//TRACER(tcr)->traceopt(6, 1);
	//TRACER(tcr)->traceopt(9, 2);
	//TRACER(tcr)->traceopt(5, 0.5);
	TRACER(tcr)->traceopt(8, 1);//small block
	TRACER(tcr)->traceopt(10, 1);// 5);
	//TRACER(tcr)->traceopt(11, 1);
	TRACER(tcr)->traceopt(9, 4);
	//TRACER(tcr)->traceopt(16, 2);
	//TRACER(tcr)->traceopt(14, 40);
	//TRACER(tcr)->traceopt(15, 2);//layer normal
	//TRACER(tcr)->traceopt(21, 1);//어텐션 수행
	//TRACER(tcr)->traceopt(16, 3);//최종 인코딩에서만 어텐션 수행
	//TRACER(tcr)->traceopt(12, 64);
	//TRACER(tcr)->traceopt(17, 1);
	//TRACER(tcr)->traceopt(18, 100);//prelu
	//TRACER(tcr)->traceopt(20, 3);//by pass print
	//TRACER(tcr)->traceopt(22, 1);//optimizier
	//TRACER(tcr)->traceopt(23, 0.1);//learnning rate
	//TRACER(tcr)->traceopt(7, 1);
	//TRACER(tcr)->traceopt(51, 1);

	TRACER(tcr)->traceopt(53, 2);//1d
	TRACER(tcr)->traceopt(42, 1);
	//TRACER(tcr)->traceopt(52, 1);
	TRACER(tcr)->traceopt(43, 16);
	TRACER(tcr)->traceopt(44, 16);
	//TRACER(tcr)->traceopt(54, 0.1);
	a = flux(tcr, { -1, x_seq_len, x_seq_len, feat_sz }, tfloat, variable);
	//a->arange(-1);
	a2 = flux(tcr, { -1, y_seq_len, y_seq_len, feat_sz }, tfloat, variable);
	
	Impulse im(tcr);
	im.trainStep(3000);
	Dynagen dgen(T_DG_TRANSLATE, &im, a, a2, hidden_sz, batch_sz * x_seq_len * x_seq_len, 
		batch_sz * y_seq_len * y_seq_len, 16, ACTF2_PRELU);
	//Dynagen dgen(T_DG_TRANSLATE, &im, a, a2, hidden_sz, 0, 0, 0);
	Flux *loss;
	intt rv = 1;
	for(intt i = 1;i < 100000; i++) {
		//tcr->printWeight();
		if(rv) {
			a->feedf(ax);
			a2->feedf(ay);
			loss = im.train(&rv);
		}
		//loss->printo();
		//dgen.dgenloss[0]->printo();
		if(rv == 0) {
		//if(i % 3 == 0) {
			printf("----------- %d ---------\n", i);
			printf("\n");
			printf("\n");
			printf("\n");
			printf("--------------------\n");
			a->feedf(ax);
			a2->feedf(ay);
			c = im.predict(&loss);
			//c = dgen.predictv(&loss);
			loss->printo();
			c->printo(2, 10);
		}
	}
	printf("ddd\n");
}
void dynagen_test1d_mse()
{
	Tracer *tcr = trace(0);
	Flux *a, *a2, *b, *c, *w, *t, *aa, *ax, *ay;
	intt x_seq_len = 256;
	intt y_seq_len = 256;
	intt batch_sz = 5;
	intx feat_sz = 1;//discret인 경우 1로 해야 함.
	intt hidden_sz = 32;
	printf("1dmse 555\n");
	ax = flux(tcr, { batch_sz, x_seq_len, feat_sz }, tfloat, variable);
	ax->arange(-1);
	ax->stdnormal();
	ax->printo(2, 10);

	TRACER(tcr)->traceopt(1, 1);
	TRACER(tcr)->traceopt(8, 1);//small block
	TRACER(tcr)->traceopt(10, 1);
	TRACER(tcr)->traceopt(9, 4);
	TRACER(tcr)->traceopt(53, 1);//1d
	TRACER(tcr)->traceopt(42, 1);
	//TRACER(tcr)->traceopt(52, 1);
	TRACER(tcr)->traceopt(43, 16);
	TRACER(tcr)->traceopt(44, 16);

	a = flux(tcr, { -1, x_seq_len, feat_sz }, tfloat, variable);
	a->arange(-1);
	a2 = flux(tcr, { -1, y_seq_len, feat_sz }, tfloat, variable);

	Impulse im(tcr);
	Dynagen dgen(T_DG_TRANSLATE, &im, a, a2, hidden_sz, 0, 0, 0);
	Flux *loss;
	intt rv = 1;
	for(intt i = 1; i < 100000; i++) {
		//tcr->printWeight();
		if(rv) {
			a->feedf(ax);
			a2->feedf(ax);
			loss = im.train(&rv);
		}
		//loss->printo();
		//dgen.dgenloss[0]->printo();
		if(rv == 0) {
			//if(i % 10 == 0) {
			printf("----------- %d ---------\n", i);
			printf("\n");
			printf("\n");
			printf("\n");
			printf("--------------------\n");
			a->feedf(ax);
			a2->feedf(ax);
			c = im.predict(&loss);
			//c = dgen.predictv(&loss);
			loss->printo();
			c->printo(2, 10);
		}
	}
	printf("ddd\n");
}
void dynagen_test1d()
{
	Tracer *tcr = trace(0);
	Flux *a, *a2, *b, *c, *w, *t, *aa, *ax, *ay;
	intt x_seq_len = 256;
	intt y_seq_len = 256;
	intt batch_sz = 5;
	intx feat_sz = 1;//discret인 경우 1로 해야 함.
	intt hidden_sz = 32;
	printf("1d 777\n");
	ax = flux(tcr, { batch_sz, x_seq_len, feat_sz }, tfloat, variable);
	ax->arange(-1);
	ay = flux(tcr, { batch_sz, y_seq_len, feat_sz }, tfloat, variable);
	ay->arange(-1);

	ax->printo(2, 10);

	TRACER(tcr)->traceopt(1, 1);
	TRACER(tcr)->traceopt(8, 1);//small block
	TRACER(tcr)->traceopt(10, 1);
	TRACER(tcr)->traceopt(9, 4);
	TRACER(tcr)->traceopt(53, 1);//1d
	TRACER(tcr)->traceopt(42, 1);
	TRACER(tcr)->traceopt(52, 1);
	TRACER(tcr)->traceopt(43, 16);
	TRACER(tcr)->traceopt(44, 16);

	a = flux(tcr, { -1, x_seq_len, feat_sz }, tfloat, variable);
	a->arange(-1);
	a2 = flux(tcr, { -1, y_seq_len, feat_sz }, tfloat, variable);

	Impulse im(tcr);
	im.trainStep(100000);
	Dynagen dgen(T_DG_TRANSLATE, &im, a, a2, hidden_sz, batch_sz * x_seq_len, batch_sz * y_seq_len, 16);
	Flux *loss;
	intt rv = 1;
	for(intt i = 1;i < 100000; i++) {
		//tcr->printWeight();
		if(rv) {
			a->feedf(ax);
			a2->feedf(ay);
			loss = im.train(&rv);
		}
		//loss->printo();
		//dgen.dgenloss[0]->printo();
		if(rv == 0) {
		//if(i % 10 == 0) {
			printf("----------- %d ---------\n", i);
			printf("\n");
			printf("\n");
			printf("\n");
			printf("--------------------\n");
			a->feedf(ax);
			a2->feedf(ay);
			c = im.predict(&loss);
			//c = dgen.predictv(&loss);
			loss->printo();
			c->printo(2, 10);
		}
	}
	printf("ddd\n");
}
void all(void)
{
	printf("$$$$$$$$$$$$$$$$$$$ kk\n");
	kk();
	printf("$$$$$$$$$$$$$$$$$$$ gg\n");
	gg();
	printf("$$$$$$$$$$$$$$$$$$$ cc\n");
	cc();
	printf("$$$$$$$$$$$$$$$$$$$ dd\n");
	dd();
	printf("$$$$$$$$$$$$$$$$$$$ ee\n");
	ee();
	printf("$$$$$$$$$$$$$$$$$$$ aa\n");
	aa();
	printf("$$$$$$$$$$$$$$$$$$$ bb\n");
	bb();
	printf("$$$$$$$$$$$$$$$$$$$ ii\n");
	ii();
	printf("$$$$$$$$$$$$$$$$$$$ jj\n");
	jj();
	printf("$$$$$$$$$$$$$$$$$$$ mm\n");
	mm();
	printf("$$$$$$$$$$$$$$$$$$$ nn\n");
	nn();
}
#include "cydrome/impulse.h"
/*
void yy(void)
{
	Tracer *tcr = trace(1);
	floatt s2[10000][2];

	Flux *s1 = flux(tcr, { 10000 }, tfloat, variable);
	s1->randn(0, 1);

	Flux *aa = flux(tcr, { 10000 }, tfloat, variable);
	aa->randn(0, 1);
	
	floatt *p1 = (floatt *)s1->begin_p();
	floatt *p2 = (floatt *)aa->begin_p();
	for(intt i = 0;i < 10000;) {//중복데이터 설정.
		for(intt j = (intt)(*p2++ * 10);j > 0 && i < 10000; j--, i++) {
			*(p1 + i) = *p2;
		}
	}
	//BaseQSort<floatt> bsort(rsc::cydrome->parallels, nullx);//오름차순
	BaseQSort<floatt> bsort(rsc::cydrome->parallels, nullx, 32, 0);//내림차순
	QSort<floatt> *ts = (QSort<floatt> *)bsort.getTeles();

	floatt *sp = (floatt *)s1->begin_p();
	//floatt v = -100;//오름차순
	floatt v = 100;//내림차순
	ts->settings(0, 10000, &s2[0][0], sp);
	ts->rsort();
	unit lap = xucurrenttime();
	ts->wsort();
	printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
	printf("xx %d\n", ts->imaxMember);
	for(intt i = 0;i < 10000; i++) {
		//if(s2[i][0] < v) {//오름차순
		if(s2[i][0] > v) {//내림차순
			printf("inverse: %f %d\n", v, (intt)s2[i][1]);
			break;
		}
		if(s2[i][0] == v) printf("dup: %f %d\n", v, (intt)s2[i][1]);
		else printf("%f %d\n", s2[i][0], (intt)s2[i][1]);
		v = s2[i][0];
		if(v != *(sp + (intt)s2[i][1])) {
			printf("error %f\n", *(sp + (intt)s2[i][1]));
			break;
		}
	}
}
*/
#define SORT_SIZE	60000
thrret _yy(thrarg arg)
{
	intx tid = (intx)arg;
	Tracer *tcr = trace(1);
	floatt s2[SORT_SIZE][2];

	Flux *s1 = flux(tcr, { SORT_SIZE }, tfloat, variable);
	s1->randn(0, 1);

	Flux *aa = flux(tcr, { SORT_SIZE }, tfloat, variable);

	//BaseQSort<floatt> bsort(rsc::cydrome->parallels, nullx);//오름차순
	BaseQSort<floatt> bsort(rsc::cydrome->parallels, nullx, 32, 0);//내림차순

	while(1) {
		aa->randn(0, 1);
		floatt *p1 = (floatt *)s1->begin_p();
		floatt *p2 = (floatt *)aa->begin_p();
		for(intt i = 0;i < SORT_SIZE;) {//중복데이터 설정.
			for(intt j = (intt)(*p2++ * 10);j > 0 && i < SORT_SIZE; j--, i++) {
				*(p1 + i) = *p2;
			}
		}
		TeleSort *ts = bsort.getTeles();
		floatt *sp = (floatt *)s1->begin_p();
		//floatt v = -100;//오름차순
		floatt v = 100;//내림차순
		ts->settings(0, SORT_SIZE, &s2[0][0], sp);
		ts->rsort();
		unit lap = xucurrenttime();
		ts->wsort();
		printf("elasp lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
		printf("xx %d %d\n", ts->imaxMember, ts->sortProsTs->iwaits);
		for(intt i = 0;i < SORT_SIZE; i++) {
			//if(s2[i][0] < v) {//오름차순
			if(s2[i][0] > v) {//내림차순
				printf("inverse: %f %f\n", v, s2[i][0]);
				exit(1);
				break;
			}
			//if(s2[i][0] == v) printf("dup: %f %d\n", v, (intt)s2[i][1]);
			//else printf("%f %d\n", s2[i][0], (intt)s2[i][1]);
			v = s2[i][0];
			if(v != *(sp + (intt)s2[i][1])) {
				printf("error %f\n", *(sp + (intt)s2[i][1]));
				exit(1);
				break;
			}
		}
		ts->doneTeles();
	}
}
void yy(void)
{
	xthread_create((void *)_yy, (void *)1);
	xthread_create((void *)_yy, (void *)2);
	xthread_create((void *)_yy, (void *)3);
	_yy((void *)4);
}
struct mgt {
	float *o, *a, *b, *bb;
};
extern void multi_gpu_test(void *p);
extern void CudaDevSet(intt gid);
thrret thr_multi_gpu(thrarg arg)
{
	multi_gpu_test((void *)arg);
	return 0;
}
void t1(void)
{
	/*
	Tracer *tcr = trace(1);
	Flux *a, *a2, *b, *c, *w, *t, *aa, *ax, *ay;
	intt x_seq_len = 16;//1//56;//32;
	intt y_seq_len = 16;//1;//7;//32;
	intt batch_sz = 2;
	intx feat_sz = 32;//discret인 경우 1로 해야 함.
	intt hidden_sz = 32;
	printf("2d***");

	//tcr->modeset(1);
	ax = flux(tcr, { batch_sz, x_seq_len, x_seq_len, feat_sz }, tfloat, variable);
	ax->arange(-1);
	a = ax->layer_normal();
	*/
	/*
	vector<int> v1{ 1,2,3 };
	vector<int> l;

	l.insert(l.begin(), v1.begin(), v1.end());
	l.push_back(6);
	l.push_back(7);
	*/
	/*
	Tracer *tcr = trace(1);
	Flux *a = flux(tcr, { 3, 16, 2 }, tfloat, variable);
	a->fill(0.0);
	Flux *b = flux(tcr, { 3, 4, 2 }, tfloat, variable);//token
	b->arange(-1);

	a->howrite(b, 0, 3);
	a->printo(2);
	a->howrite(b, 8, 2);
	a->printo(2);
	a->howrite(b, 14);
	a->printo(2);
	*/
	/*
	Tracer *tcr = trace(1);
	auto aa = flux(tcr, { 3, 4, 3 }, tfloat, variable);
	//auto aa = flux(tcr, { 3, 4, 1 }, tfloat, variable);
	//aa->randn(0, 1);
	aa->arange(-1);
	aa = *aa + 1.0;
	aa->printo();
	aa->xnormal(0);// , 1);
	aa->printo();
	aa->xnormal(1);// , 1);
	aa->printo();
	*/
	printf("777\n");
	/*CudaDevSet(1);
	struct mgt mp;
	mp.o = (float *)malloc(sizeof(float) * 16 * 1280 * 720);
	mp.bb = (float *)malloc(sizeof(float) * 16 * 1280 * 720);
	cudaMalloc(&mp.a, sizeof(float) * 16 * 1280 * 720);
	cudaMalloc(&mp.b, sizeof(float) * 16 * 1280 * 720);
	xthread_create((void *)thr_multi_gpu, (void *)&mp);
	printf("zzz\n");
	xsleep(100);
	return;*/
	Tracer *tcr = trace(1);
	tcr->npset(1000);
	TRACER(tcr)->traceopt(8, 1);//small block
	tcr->setgpudev(1);
	/*
	Flux *a, *b, *c;

	a = flux(tcr, { 2, 3 }, tfloat, variable);
	a->arange(2 * 3)->printo();
	b = flux(tcr, { 3,2 }, tfloat, variable);
	b->arange(3 * 2)->printo();
	c = a->dot(b, { {1}, {0} }, 0);
	c->printo();
	*/
	/*
	//auto aa = flux(tcr, { 3, 4, 3 }, tfloat, variable);
	auto aa = flux(tcr, { 3, 4, 1 }, tfloat, variable);
	//aa->randn(0, 1);
	aa->arange(-1);
	aa = *aa + 1.0;
	aa->printo();
	auto bb = flux(aa, variable);
	aa->xnormal(0, 1);
	aa->printo();
	aa->xrnormal(bb, 1);
	aa->printo();
	*/
	/*
	//tcr->modeset(1);
	auto inp = flux(tcr, { 3, 2, 4, 12 }, tfloat, constant);
	inp->fill(0.0);
	auto enpos = flux(tcr, { 2, 4, 12 }, tfloat, constant);
	enpos->fill(0.0);
	enpos->sinpos(2*4);
	enpos->printo();
	inp->printo();
	inp = *inp + *enpos;//sinuosid positional
	inp->printo();
	*/
	/*
	auto aa = flux(tcr, { 2, 1, 2, 4, 8 }, tfloat, variable);
	aa->arange(-1);
	aa = aa->transpose({ 0, 1, 3, 2, 4 });
	aa->printo();
	*/
	
	auto a = flux(tcr, { 2,32,1 }, tfloat, trainable);
	a = a->reshape({ -1, 4, 8, 1 });
	a->arange(-1);
	a->printo(2);
	a = a->slice({ { }, {}, {5, -1} });
	a->printo(2);
	
	/*
	auto a = flux(tcr, { 4, 1, 3 }, tfloat, variable);
	a->fill(1.0)->printo();
	auto b = flux(tcr, { 1, 3 }, tfloat, variable);
	b->arange(1 * 3)->printo();
	auto c = a->mul(b);
	c->printo();
	*/
}
void ff(void)
{

	for(intt i = 0;; i++) {
		all();
	}
}
#include <iomanip>
void mpmain(intt sz)
{
	BoostMemput(0, 2);
	//nn2();
	//cnet_not_embedding_mean_square_test();
	//cnet_embedding_cross_entropy_test2();
	//t1();
	//soft_argmax_test();
	//dual_image2seq_test(sz);
	/*if(sz == 1) dynagen_test1d();
	else if(sz == 2) dynagen_test2d();
	else if(sz == 3) dynagen_test1d_mse();
	else if(sz == 4) dynagen_test2d_mse();*/
	//cnet_not_embedding_mean_square_convolution_test();
	//stratus_embedding_cross_entropy_convolution_test(sz);
	//algol_embedding_cross_entropy_convolution_test(sz);
	//cnet_embedding_cross_entropy_convolution_test(sz);//ff();//performance_test(sz);//ss();//rr();//tt();//
	cnet_embedding_cross_entropy_dual_test(sz);
	//cnet_embedding_cross_entropy_dual_dgen1d_test(sz);
	//cnet_mse_dual_dgen1d_16y_test();
}