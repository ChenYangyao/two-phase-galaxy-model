<div align="center">
  <img width="400px" src="https://raw.githubusercontent.com/ChenYangyao/two-phase-galaxy-model/master/site-info/logo-small.png"/>
</div>

# Two Phase Galaxy Model

A two-phase scenario of galaxy formation and its semi-analytic implementation.

## Publications

- (Paper-I) A two-phase model of galaxy formation: I. The growth of galaxies and supermassive black holes. *Houjun Mo, Yangyao Chen, and Huiyuan Wang, 2023* ([arxiv](https://arxiv.org/abs/2311.05030), [ads](https://ui.adsabs.harvard.edu/abs/arXiv:2311.05030)).
- (Paper-II) A two-phase model of galaxy formation: II. The size-mass relation of dynamically hot galaxies.
*Yangyao Chen, Houjun Mo, and Huiyuan Wang, 2023* ([arxiv](https://arxiv.org/abs/2311.11713), [ads](https://ui.adsabs.harvard.edu/abs/arXiv:2311.11713)).
- (Paper-III) A two-phase model of galaxy formation: III. The formation of globular clusters. *Yangyao Chen, Houjun Mo, and Huiyuan Wang, 2024* ([arxiv](http://arxiv.org/abs/2405.18735), [ads](https://ui.adsabs.harvard.edu/abs/arXiv:2405.18735)).

## Highlights

### The phase diagram of galaxy formation and morphology transformation.

<div align="center">
  <img width="550px" src="https://raw.githubusercontent.com/ChenYangyao/two-phase-galaxy-model/master/site-info/highlights/phase-space-diagram.jpg"/>
  <div>See Paper-I. The quadrant diagram showing four combinations of 
    halo assembly rate and the importance of angular momentum in supporting gas.</div>
</div>

### The formation of dynamically hot systems (bulges, ellipticals, etc.)

<div align="center">
  <img width="550px" src="https://raw.githubusercontent.com/ChenYangyao/two-phase-galaxy-model/master/site-info/highlights/dynamical-hot-stage.jpg"/>
  <div>See Paper-I. The formation of dynamically hot gas/stellar system in the fast stage of dark matter halo. </div>
</div>

### A angular-momentum-limited scenario of SMBH growth

<div align="center">
  <img width="550px" src="https://raw.githubusercontent.com/ChenYangyao/two-phase-galaxy-model/master/site-info/highlights/smbh-growth-am.jpg"/>
  <div>See Paper-I. A schematic diagram showing the distribution of specific angular momentum (sAM)
        for gas clouds within a halo. The turbulent motion of gas clouds, 
        driven by fast accretion, yields a broad and uniform distribution 
        (red curve) of sAM. The fraction of gas accreted by the 
        SMBH (gray shaded area) is determined by the maximum capturing 
        angular momentum. Subsequently, as the driving force of
        turbulent motion diminishes, gas mixing becomes significant, 
        leading to the emergence of an angular momentum barrier and 
        preventing gas accretion (blue curve). This diagram shows the 
        mechanism underlying the formation of dynamically hot systems, 
        the accretion scenario of SMBH within turbulent gas clouds, 
        and the transition to dynamically cold systems. </div>
</div>

### The two-channels of GC formation

<div align="center">
  <img width="850px" src="https://raw.githubusercontent.com/ChenYangyao/two-phase-galaxy-model/master/site-info/highlights/gc-channels.jpg"/>
  <div>See Paper-III. The criteria for the formation of Pop-I (red, metal-rich) and Pop-II (blue, metal-poor) GCs. </div>
</div>

### Cosmic structures traced by GC clustering over > 7 orders of magnitude in spatial extent.

<div align="center">
  <img width="850px" src="https://raw.githubusercontent.com/ChenYangyao/two-phase-galaxy-model/master/site-info/highlights/tpcf.jpg"/>
  <div>See Paper-III. Two-point auto-correlation functions of 
        GCs at z=0 predicted by the model in this paper.</div>
</div>

## Supplementary Material of the Publications

- Paper-I: [figures](publications/Paper-I/figures) in the pdf format.
- Paper-II: [figures](publications/Paper-II/figures) in the pdf format.
- Paper-III: [figures](publications/Paper-III/figures) in the pdf format.

## TODO List

- Code for the model. Documentation and code samples.
- Raw data for figures.
- Catalogs of halo assembly histories and modeled galaxies/SMBHs. 

## Acknowledgements

Here we acknowledge the following projects and/or people that helped in the development of 
this code repository. Projects and/or people that are relevant to individual papers
were included in the papers.

Yangyao thanks [Kai Wang](https://www.kosmoswalker.com/) for his enduring 
companionship and support :wink:.

The empirical model 
[UniverseMachine](https://bitbucket.org/pbehroozi/universemachine/src/main/) for halo-galaxy connection,
and [TRINITY](https://github.com/HaowenZhang/TRINITY) for halo-galaxy-SMBH coevolution,
have inspired our work for model design and implementation. Their comprehensive
collection of observational datasets also helps us for model calibration and
comparison.

The SAM of Yingtian Chen, Oleg Y. Gnedin et al. ([source code](https://github.com/ybillchen/GC_formation_model)) 
and the [catalog](https://github.com/ognedin/gc_model_mw) produced by the model have been used for comparison of our model.

The Tsinghua Astrophysics High-Performance Computing platform has been
providing the computational and data storage resources throughout the 
career development of Yangyao.