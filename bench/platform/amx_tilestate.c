// Minimal AMX tile-state preservation probe. No GEMM, no library code.
//
// Load a known pattern into a tile, burn wall-clock so the scheduler can
// preempt us, store the tile back, compare. If the kernel and hypervisor save
// and restore XTILEDATA correctly across a context switch, this can never fail.
//
// On the target VM it fails, and the tile comes back ZEROED (reset to INIT
// state). Corruption probability tracks how long the state is held, which is
// the signature of losing it on a context switch:
//
//     hold        corrupted (400 trials, 6 competing spinners)
//     0           0%
//     10 us       0.25%
//     100 us      3%
//     1 ms        16%
//     10 ms       89%
//
// implying a context switch roughly every ~7 ms, each of which destroys the
// tile file. This is a platform defect, not an application bug: it reproduces
// with nothing but _tile_loadd, a busy-wait and _tile_stored.
//
// Consequence for any AMX GEMM: an accumulator held in a tile across a k-loop
// silently loses everything accumulated before the switch. Results are wrong,
// intermittently, and only under CPU contention -- which is to say, only in
// production. See JOURNAL.md section 8.
//
//   gcc -O2 -march=sapphirerapids -mamx-int8 -mamx-tile -o amx_tilestate amx_tilestate.c
//   ./amx_tilestate [trials] [hold_seconds]
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <immintrin.h>
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18
typedef struct { uint8_t p,s,r[14]; uint16_t colsb[16]; uint8_t rows[16]; } cfg_t;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(int argc,char**argv){
  if(syscall(SYS_arch_prctl,ARCH_REQ_XCOMP_PERM,XFEATURE_XTILEDATA)!=0){puts("AMX perm failed");return 1;}
  cfg_t c; memset(&c,0,sizeof c); c.p=1;
  for(int i=0;i<8;i++){c.rows[i]=16;c.colsb[i]=64;}
  _tile_loadconfig(&c);
  int trials = argc>1?atoi(argv[1]):200;
  double hold = argc>2?atof(argv[2]):0.002;   // seconds to hold state
  _Alignas(64) int8_t pat[1024], out[1024];
  for(int i=0;i<1024;i++) pat[i]=(int8_t)(i*37+11);
  int bad=0;
  volatile double sink=0;
  for(int t=0;t<trials;t++){
    _tile_loadd(0,pat,64);
    double t0=now(); while(now()-t0<hold) sink+=1.0;   // preemptible window
    memset(out,0,sizeof out);
    _tile_stored(0,out,64);
    if(memcmp(out,pat,1024)){
      bad++;
      if(bad==1){ int j=0; while(j<1024&&out[j]==pat[j])j++;
        printf("  first corrupt byte at %d: expected %d got %d\n",j,pat[j],out[j]); }
    }
  }
  (void)sink;
  _tile_release();
  printf("tile-state trials=%d hold=%.3fs corrupted=%d (%.0f%%)  %s\n",
         trials,hold,bad,100.0*bad/trials, bad? "TILE STATE NOT PRESERVED":"ok");
  return 0;
}
