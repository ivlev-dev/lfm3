#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
static int exp_bit(int32_t xq){int32_t y,n,f,p;
 if(xq<-16*65536)return 0; if(xq>0)xq=0;
 y=(int32_t)(((int64_t)xq*94548)>>16); n=y>>16; f=(y-(n<<16))>>2;
 p=3637; p=(int32_t)(((int64_t)f*p)>>14)+15742; p=(int32_t)(((int64_t)f*p)>>14)+45426;
 p=(int32_t)(((int64_t)f*p)>>14)+65536; n=-n; if(n>=15)return 0; return (int32_t)((p>>n)>>8);}
static const int16_t tab[33]={255,155,94,57,35,21,13,8,5,3,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
static int exp_pwl(int32_t xq){int32_t u; if(xq>=0)return 255; if(xq<-16*65536)return 0; u=(-xq)>>15; return tab[u>32?32:u];}
int main(void){int i; double eb=0,ep=0,wb=0,wp=0; double sb=0,sp=0;
 for(i=0;i<=200000;i++){double x=-16.0+16.0*i/200000.0; int32_t xq=(int32_t)llround(x*65536.0);
  double ref=exp(x), vb=exp_bit(xq)/255.0, vp=exp_pwl(xq)/255.0;
  if(ref>1e-4){double r1=fabs(vb-ref)/ref, r2=fabs(vp-ref)/ref; if(r1>eb){eb=r1;wb=x;} if(r2>ep){ep=r2;wp=x;}}
  sb+=fabs(vb-ref); sp+=fabs(vp-ref);}
 printf("bit-trick : max rel %.3e (x=%.2f)  mean abs %.3e\n", eb, wb, sb/200001);
 printf("piecewise : max rel %.3e (x=%.2f)  mean abs %.3e\n", ep, wp, sp/200001);
 return 0;}
