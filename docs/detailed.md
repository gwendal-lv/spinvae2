---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults
#

layout: splash
classes: wide

---

<script type="text/javascript" src="../assets/js/howler.min.js"></script>
<script type="text/javascript" src="../assets/js/listen_detailed.js"></script>

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    jax: ["input/TeX", "output/HTML-CSS"],
    tex2jax: {
      inlineMath: [ ['$', '$'], ["\\(", "\\)"] ],
      displayMath: [ ['$$', '$$'], ["\\[", "\\]"] ],
      processEscapes: true,
      skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    }
    //,
    //displayAlign: "left",
    //displayIndent: "2em"
  });
</script>
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-MML-AM_CHTML">
</script>

<link rel="stylesheet" href="../assets/css/styles.css">

*Author: Gwendal Le Vaillant ([ISIA Lab](https://web.umons.ac.be/isia/), University of Mons)*

TODO detailed examples...


---

# Interpolation Between Presets

TODO refaire tout le texte d'into... EN précisant à quel chapitre les exemples sont liés

Preset interpolation is usually performed by computing a linear interpolation on each individual synthesis parameter.
This method is called <em>Reference (linear)</em>.

Our work introduces a preset morphing method based on the <em>SPINVAE-2</em> model. First, the start and end presets, $$ \mathbf{u}^{(n)} $$ and $$ \mathbf{u}^{(m)} $$, are encoded into latent vectors $$ \mathbf{z}^{(n)} $$ and $$ \mathbf{z}^{(m)} $$.
Second, a linear interpolation produces a series of intermediate latent codes $\lbrace \mathbf{z}[t], t \in [1, T] \rbrace$ vectors, where $\mathbf{z}[1] = \mathbf{z}^{(n)}$ and $\mathbf{z}[T] = \mathbf{z}^{(m)}$.
Third, the preset morphing is finally obtained by decoding each $\mathbf{z}[t]$ into an intermediate preset.

Examples below use $$T = 9$$ interpolation steps for both methods.


### Interpolation example 1


<div class="figure">
    <table>
        <tr>
            <th></th>
            <th colspan="9" >
                Start preset, step 1/9: "E.Piano 23"<br/>
                End preset, step 9/9: "B3 Organ 3"
            </th>
        </tr>
        <tr>
            <th></th>
            <td>Step 1/9</td>
            <td>Step 2/9</td>
            <td>Step 3/9</td>
            <td>Step 4/9</td>
            <td>Step 5/9</td>
            <td>Step 6/9</td>
            <td>Step 7/9</td>
            <td>Step 8/9</td>
            <td>Step 9/9</td>
        </tr>
        <tr>  <!-- LINEAR PARAMETRIC interp -->
            <th colspan="10" >Linear parametric preset interpolation (linearity = -0.98 ; smoothness = -60.5)</th>  
        </tr>
        <tr  class="no-bottom-border"> 
            <th scope="row">
                <button type="button" id="playSequence3" onclick="onPlaySequenceButtonClicked(3)">
                    <img src="../assets/svg/play.svg" class="play_button"/>  <br> Play all
                </button>
            </th>
            <td>
                <button type="button" onclick="onPlayButtonClicked(3, 0)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq3_wave0" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(3, 1)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq3_wave1" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(3, 2)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq3_wave2" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(3, 3)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq3_wave3" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(3, 4)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq3_wave4" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(3, 5)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq3_wave5" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(3, 6)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq3_wave6" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(3, 7)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq3_wave7" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(3, 8)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq3_wave8" class="soundwave"/>
            </td>
        </tr>
        <tr>
            <td colspan="10"><img src="../assets/detailed/263/linear/spec_and_features.png"/></td>
        </tr>
        <tr>  <!-- SPINVAE interp -->
            <th colspan="10" >SPINVAE preset interpolation (linearity = -0.39 ; smoothness = -39.2)</th>  
        </tr>
        <tr  class="no-bottom-border"> 
            <td></td>
            <td colspan=5>Start sound reconstruction:<br/>MFCCD = 0<br/>PEMO-Q ODG = 0</td>
            <td colspan=4 class="right_text_align">End sound reconstruction:<br/>MFCCD = 0.3<br/>PEMO-Q ODG = -0.01</td>
        </tr>
        <tr  class="no-bottom-border"> 
            <th scope="row">
                <button type="button" id="playSequence4" onclick="onPlaySequenceButtonClicked(4)">
                    <img src="../assets/svg/play.svg" class="play_button"/>  <br> Play all
                </button>
            </th>
            <td>
                <button type="button" onclick="onPlayButtonClicked(4, 0)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq4_wave0" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(4, 1)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq4_wave1" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(4, 2)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq4_wave2" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(4, 3)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq4_wave3" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(4, 4)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq4_wave4" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(4, 5)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq4_wave5" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(4, 6)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq4_wave6" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(4, 7)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq4_wave7" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(4, 8)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq4_wave8" class="soundwave"/>
            </td>
        </tr>
        <tr>
            <td colspan="10"><img src="../assets/detailed/263/spinvae/spec_and_features.png"/></td>
        </tr>
        <tr>  <!-- SMT sound morphing interp -->
            <th colspan="10" >SMT sound morphing (linearity = -0.25 ; smoothness = -11.5)</th>  
        </tr>
        <tr  class="no-bottom-border"> 
            <td></td>
            <td colspan=5>Start sound reconstruction:<br/>MFCCD = 3.0<br/>PEMO-Q ODG = -0.68</td>
            <td colspan=4 class="right_text_align">End sound reconstruction:<br/>MFCCD = 4.0<br/>PEMO-Q ODG = -1.66</td>
        </tr>
        <tr  class="no-bottom-border"> 
            <th scope="row">
                <button type="button" id="playSequence5" onclick="onPlaySequenceButtonClicked(5)">
                    <img src="../assets/svg/play.svg" class="play_button"/>  <br> Play all
                </button>
            </th>
            <td>
                <button type="button" onclick="onPlayButtonClicked(5, 0)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq5_wave0" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(5, 1)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq5_wave1" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(5, 2)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq5_wave2" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(5, 3)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq5_wave3" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(5, 4)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq5_wave4" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(5, 5)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq5_wave5" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(5, 6)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq5_wave6" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(5, 7)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq5_wave7" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(5, 8)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq5_wave8" class="soundwave"/>
            </td>
        </tr>
        <tr>
            <td colspan="10"><img src="../assets/detailed/263/smt/spec_and_features.png"/></td>
        </tr>
    </table>
</div>






### Interpolation example 2

<div class="figure">
    <table>
        <tr>
            <th></th>
            <th colspan="9" >
                Start preset, step 1/9: "AnlgSyn.45"<br/>
                End preset, step 9/9: "ClinkieBel"
            </th>
        </tr>
        <tr>
            <th></th>
            <td>Step 1/9</td>
            <td>Step 2/9</td>
            <td>Step 3/9</td>
            <td>Step 4/9</td>
            <td>Step 5/9</td>
            <td>Step 6/9</td>
            <td>Step 7/9</td>
            <td>Step 8/9</td>
            <td>Step 9/9</td>
        </tr>
        <tr>  <!-- LINEAR PARAMETRIC interp -->
            <th colspan="10" >Linear parametric preset interpolation (linearity = -0.55 ; smoothness = -64.4)</th>  
        </tr>
        <tr  class="no-bottom-border"> 
            <th scope="row">
                <button type="button" id="playSequence0" onclick="onPlaySequenceButtonClicked(0)">
                    <img src="../assets/svg/play.svg" class="play_button"/>  <br> Play all
                </button>
            </th>
            <td>
                <button type="button" onclick="onPlayButtonClicked(0, 0)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq0_wave0" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(0, 1)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq0_wave1" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(0, 2)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq0_wave2" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(0, 3)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq0_wave3" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(0, 4)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq0_wave4" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(0, 5)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq0_wave5" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(0, 6)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq0_wave6" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(0, 7)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq0_wave7" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(0, 8)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq0_wave8" class="soundwave"/>
            </td>
        </tr>
        <tr>
            <td colspan="10"><img src="../assets/detailed/254/linear/spec_and_features.png"/></td>
        </tr>
        <tr>  <!-- SPINVAE interp -->
            <th colspan="10" >SPINVAE preset interpolation (linearity = -0.42 ; smoothness = -41.8)</th>  
        </tr>
        <tr  class="no-bottom-border"> 
            <td></td>
            <td colspan=5>Start sound reconstruction:<br/>MFCCD = 0.36<br/>PEMO-Q ODG = -0.03</td>
            <td colspan=4 class="right_text_align">End sound reconstruction:<br/>MFCCD = 0.05<br/>PEMO-Q ODG = -0.01</td>
        </tr>
        <tr  class="no-bottom-border"> 
            <th scope="row">
                <button type="button" id="playSequence1" onclick="onPlaySequenceButtonClicked(1)">
                    <img src="../assets/svg/play.svg" class="play_button"/>  <br> Play all
                </button>
            </th>
            <td>
                <button type="button" onclick="onPlayButtonClicked(1, 0)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq1_wave0" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(1, 1)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq1_wave1" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(1, 2)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq1_wave2" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(1, 3)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq1_wave3" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(1, 4)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq1_wave4" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(1, 5)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq1_wave5" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(1, 6)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq1_wave6" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(1, 7)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq1_wave7" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(1, 8)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq1_wave8" class="soundwave"/>
            </td>
        </tr>
        <tr>
            <td colspan="10"><img src="../assets/detailed/254/spinvae/spec_and_features.png"/></td>
        </tr>
        <tr>  <!-- SMT sound morphing interp -->
            <th colspan="10" >SMT sound morphing (linearity = -0.30 ; smoothness = -15.8)</th>  
        </tr>
        <tr  class="no-bottom-border"> 
            <td></td>
            <td colspan=5>Start sound reconstruction:<br/>MFCCD = 3.5<br/>PEMO-Q ODG = -0.67</td>
            <td colspan=4 class="right_text_align">End sound reconstruction:<br/>MFCCD = 5.3<br/>PEMO-Q ODG = -0.49</td>
        </tr>
        <tr  class="no-bottom-border"> 
            <th scope="row">
                <button type="button" id="playSequence2" onclick="onPlaySequenceButtonClicked(2)">
                    <img src="../assets/svg/play.svg" class="play_button"/>  <br> Play all
                </button>
            </th>
            <td>
                <button type="button" onclick="onPlayButtonClicked(2, 0)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq2_wave0" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(2, 1)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq2_wave1" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(2, 2)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq2_wave2" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(2, 3)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq2_wave3" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(2, 4)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq2_wave4" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(2, 5)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq2_wave5" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(2, 6)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq2_wave6" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(2, 7)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq2_wave7" class="soundwave"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(2, 8)"><img src="../assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="../assets/svg/soundwave.svg" id="seq2_wave8" class="soundwave"/>
            </td>
        </tr>
        <tr>
            <td colspan="10"><img src="../assets/detailed/254/smt/spec_and_features.png"/></td>
        </tr>
    </table>
</div>


