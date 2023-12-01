---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults
#

layout: splash
classes: wide

---


<script type="text/javascript" src="assets/js/howler.min.js"></script>
<script type="text/javascript" src="assets/js/listen.js"></script>

<link rel="stylesheet" href="assets/css/styles.css">

*Authors: Gwendal Le Vaillant and Thierry Dutoit ([ISIA Lab](https://web.umons.ac.be/isia/), University of Mons)*

Work in progress.

---

# Interpolation between presets

### Interpolation example 1

<div class="figure">
    <table>
        <tr>
            <th></th>
            <th>Start preset<br/>"AnlgSyn.45"</th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th>End preset<br />"ClinkieBel"</th>
        </tr>
        <tr>
            <th></th>
            <td>Step 1/7</td>
            <td>Step 2/7</td>
            <td>Step 3/7</td>
            <td>Step 4/7</td>
            <td>Step 5/7</td>
            <td>Step 6/7</td>
            <td>Step 7/7</td>
        </tr>
        <tr> <!-- SPINVAE interp -->
            <th scope="row">
                <button type="button" id="playSequence4" onclick="onPlaySequenceButtonClicked(4)">
                    <img src="assets/svg/play.svg" class="play_button"/>  <br> Play all
                </button>
                <br>  <br> SPINVAE-2 <br> &nbsp;
            </th>
            <td>
                <button type="button" onclick="onPlayButtonClicked(4, 0)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq4_wave0" class="soundwave"/><br />
                <img src="assets/interpolation/254_spinvae/spectrogram_step00.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(4, 1)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq4_wave1" class="soundwave"/><br />
                <img src="assets/interpolation/254_spinvae/spectrogram_step01.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(4, 2)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq4_wave2" class="soundwave"/><br />
                <img src="assets/interpolation/254_spinvae/spectrogram_step02.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(4, 3)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq4_wave3" class="soundwave"/><br />
                <img src="assets/interpolation/254_spinvae/spectrogram_step03.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(4, 4)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq4_wave4" class="soundwave"/><br />
                <img src="assets/interpolation/254_spinvae/spectrogram_step04.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(4, 5)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq4_wave5" class="soundwave"/><br />
                <img src="assets/interpolation/254_spinvae/spectrogram_step05.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(4, 6)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq4_wave6" class="soundwave"/><br />
                <img src="assets/interpolation/254_spinvae/spectrogram_step06.png"/>
            </td>
        </tr>
        <tr> <!-- REFERENCE interp -->
            <th scope="row">
                <button type="button" id="playSequence5" onclick="onPlaySequenceButtonClicked(5)">
                    <img src="assets/svg/play.svg" class="play_button"/>  <br> Play all
                </button>
                <br>  <br> Reference <br> (linear)
            </th>
            <td>
                <button type="button" onclick="onPlayButtonClicked(5, 0)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq5_wave0" class="soundwave"/><br />
                <img src="assets/interpolation/254_reference/spectrogram_step00.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(5, 1)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq5_wave1" class="soundwave"/><br />
                <img src="assets/interpolation/254_reference/spectrogram_step01.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(5, 2)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq5_wave2" class="soundwave"/><br />
                <img src="assets/interpolation/254_reference/spectrogram_step02.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(5, 3)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq5_wave3" class="soundwave"/><br />
                <img src="assets/interpolation/254_reference/spectrogram_step03.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(5, 4)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq5_wave4" class="soundwave"/><br />
                <img src="assets/interpolation/254_reference/spectrogram_step04.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(5, 5)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq5_wave5" class="soundwave"/><br />
                <img src="assets/interpolation/254_reference/spectrogram_step05.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(5, 6)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq5_wave6" class="soundwave"/><br />
                <img src="assets/interpolation/254_reference/spectrogram_step06.png"/>
            </td>
        </tr>
    </table>
</div>

### Interpolation example 2

<div class="figure">
    <table>
        <tr>
            <th></th>
            <th>Start preset<br/>"E.Piano 23"</th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th>End preset<br />"B3 Organ 3"</th>
        </tr>
        <tr>
            <th></th>
            <td>Step 1/7</td>
            <td>Step 2/7</td>
            <td>Step 3/7</td>
            <td>Step 4/7</td>
            <td>Step 5/7</td>
            <td>Step 6/7</td>
            <td>Step 7/7</td>
        </tr>
        <tr> <!-- SPINVAE interp -->
            <th scope="row">
                <button type="button" onclick="onPlaySequenceButtonClicked(6)">
                    <img src="assets/svg/play.svg" class="play_button"/>  <br> Play all
                </button>
                <br>  <br> SPINVAE-2 <br> &nbsp;
            </th>
            <td>
                <button type="button" onclick="onPlayButtonClicked(6, 0)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq6_wave0" class="soundwave"/><br />
                <img src="assets/interpolation/263_spinvae/spectrogram_step00.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(6, 1)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq6_wave1" class="soundwave"/><br />
                <img src="assets/interpolation/263_spinvae/spectrogram_step01.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(6, 2)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq6_wave2" class="soundwave"/><br />
                <img src="assets/interpolation/263_spinvae/spectrogram_step02.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(6, 3)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq6_wave3" class="soundwave"/><br />
                <img src="assets/interpolation/263_spinvae/spectrogram_step03.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(6, 4)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq6_wave4" class="soundwave"/><br />
                <img src="assets/interpolation/263_spinvae/spectrogram_step04.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(6, 5)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq6_wave5" class="soundwave"/><br />
                <img src="assets/interpolation/263_spinvae/spectrogram_step05.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(6, 6)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq6_wave6" class="soundwave"/><br />
                <img src="assets/interpolation/263_spinvae/spectrogram_step06.png"/>
            </td>
        </tr>
        <tr> <!-- REFERENCE interp -->
            <th scope="row">
                <button type="button" onclick="onPlaySequenceButtonClicked(7)">
                    <img src="assets/svg/play.svg" class="play_button"/>  <br> Play all
                </button>
                <br>  <br> Reference <br> (linear)
            </th>
            <td>
                <button type="button" onclick="onPlayButtonClicked(7, 0)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq7_wave0" class="soundwave"/><br />
                <img src="assets/interpolation/263_reference/spectrogram_step00.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(7, 1)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq7_wave1" class="soundwave"/><br />
                <img src="assets/interpolation/263_reference/spectrogram_step01.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(7, 2)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq7_wave2" class="soundwave"/><br />
                <img src="assets/interpolation/263_reference/spectrogram_step02.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(7, 3)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq7_wave3" class="soundwave"/><br />
                <img src="assets/interpolation/263_reference/spectrogram_step03.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(7, 4)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq7_wave4" class="soundwave"/><br />
                <img src="assets/interpolation/263_reference/spectrogram_step04.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(7, 5)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq7_wave5" class="soundwave"/><br />
                <img src="assets/interpolation/263_reference/spectrogram_step05.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(7, 6)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq7_wave6" class="soundwave"/><br />
                <img src="assets/interpolation/263_reference/spectrogram_step06.png"/>
            </td>
        </tr>
    </table>
</div>


### Interpolation example 3

<div class="figure">
    <table>
        <tr>
            <th></th>
            <th>Start preset<br/>"WindEns2Ed"</th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th>End preset<br />"HARD ROADS"</th>
        </tr>
        <tr>
            <th></th>
            <td>Step 1/7</td>
            <td>Step 2/7</td>
            <td>Step 3/7</td>
            <td>Step 4/7</td>
            <td>Step 5/7</td>
            <td>Step 6/7</td>
            <td>Step 7/7</td>
        </tr>
        <tr> <!-- SPINVAE interp -->
            <th scope="row">
                <button type="button" onclick="onPlaySequenceButtonClicked(0)">
                    <img src="assets/svg/play.svg" class="play_button"/>  <br> Play all
                </button>
                <br>  <br> SPINVAE-2 <br> &nbsp;
            </th>
            <td>
                <button type="button" onclick="onPlayButtonClicked(0, 0)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq0_wave0" class="soundwave"/><br />
                <img src="assets/interpolation/135_spinvae/spectrogram_step00.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(0, 1)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq0_wave1" class="soundwave"/><br />
                <img src="assets/interpolation/135_spinvae/spectrogram_step01.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(0, 2)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq0_wave2" class="soundwave"/><br />
                <img src="assets/interpolation/135_spinvae/spectrogram_step02.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(0, 3)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq0_wave3" class="soundwave"/><br />
                <img src="assets/interpolation/135_spinvae/spectrogram_step03.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(0, 4)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq0_wave4" class="soundwave"/><br />
                <img src="assets/interpolation/135_spinvae/spectrogram_step04.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(0, 5)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq0_wave5" class="soundwave"/><br />
                <img src="assets/interpolation/135_spinvae/spectrogram_step05.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(0, 6)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq0_wave6" class="soundwave"/><br />
                <img src="assets/interpolation/135_spinvae/spectrogram_step06.png"/>
            </td>
        </tr>
        <tr> <!-- REFERENCE interp -->
            <th scope="row">
                <button type="button" id="playSequence1" onclick="onPlaySequenceButtonClicked(1)">
                    <img src="assets/svg/play.svg" class="play_button"/>  <br> Play all
                </button>
                <br>  <br> Reference <br> (linear)
            </th>
            <td>
                <button type="button" onclick="onPlayButtonClicked(1, 0)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq1_wave0" class="soundwave"/><br />
                <img src="assets/interpolation/135_reference/spectrogram_step00.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(1, 1)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq1_wave1" class="soundwave"/><br />
                <img src="assets/interpolation/135_reference/spectrogram_step01.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(1, 2)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq1_wave2" class="soundwave"/><br />
                <img src="assets/interpolation/135_reference/spectrogram_step02.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(1, 3)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq1_wave3" class="soundwave"/><br />
                <img src="assets/interpolation/135_reference/spectrogram_step03.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(1, 4)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq1_wave4" class="soundwave"/><br />
                <img src="assets/interpolation/135_reference/spectrogram_step04.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(1, 5)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq1_wave5" class="soundwave"/><br />
                <img src="assets/interpolation/135_reference/spectrogram_step05.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(1, 6)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq1_wave6" class="soundwave"/><br />
                <img src="assets/interpolation/135_reference/spectrogram_step06.png"/>
            </td>
        </tr>
    </table>
</div>


### Interpolation example 4

<div class="figure">
    <table>
        <tr>
            <th></th>
            <th>Start preset<br/>"ABU ASHRAM"</th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th>End preset<br />"STARRY 1"</th>
        </tr>
        <tr>
            <th></th>
            <td>Step 1/7</td>
            <td>Step 2/7</td>
            <td>Step 3/7</td>
            <td>Step 4/7</td>
            <td>Step 5/7</td>
            <td>Step 6/7</td>
            <td>Step 7/7</td>
        </tr>
        <tr> <!-- SPINVAE interp -->
            <th scope="row">
                <button type="button" id="playSequence2" onclick="onPlaySequenceButtonClicked(2)">
                    <img src="assets/svg/play.svg" class="play_button"/>  <br> Play all
                </button>
                <br>  <br> SPINVAE-2 <br> &nbsp;
            </th>
            <td>
                <button type="button" onclick="onPlayButtonClicked(2, 0)" id="playSeq0Sound0">
                    <img src="assets/svg/play.svg" class="play_button"/>
                </button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq2_wave0" class="soundwave"/><br />
                <img src="assets/interpolation/6_spinvae/spectrogram_step00.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(2, 1)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq2_wave1" class="soundwave"/><br />
                <img src="assets/interpolation/6_spinvae/spectrogram_step01.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(2, 2)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq2_wave2" class="soundwave"/><br />
                <img src="assets/interpolation/6_spinvae/spectrogram_step02.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(2, 3)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq2_wave3" class="soundwave"/><br />
                <img src="assets/interpolation/6_spinvae/spectrogram_step03.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(2, 4)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq2_wave4" class="soundwave"/><br />
                <img src="assets/interpolation/6_spinvae/spectrogram_step04.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(2, 5)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq2_wave5" class="soundwave"/><br />
                <img src="assets/interpolation/6_spinvae/spectrogram_step05.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(2, 6)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq2_wave6" class="soundwave"/><br />
                <img src="assets/interpolation/6_spinvae/spectrogram_step06.png"/>
            </td>
        </tr>
        <tr> <!-- REFERENCE interp -->
            <th scope="row">
                <button type="button" id="playSequence3" onclick="onPlaySequenceButtonClicked(3)">
                    <img src="assets/svg/play.svg" class="play_button"/>  <br> Play all
                </button>
                <br>  <br> Reference <br> (linear)
            </th>
            <td>
                <button type="button" onclick="onPlayButtonClicked(3, 0)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq3_wave0" class="soundwave"/><br />
                <img src="assets/interpolation/6_reference/spectrogram_step00.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(3, 1)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq3_wave1" class="soundwave"/><br />
                <img src="assets/interpolation/6_reference/spectrogram_step01.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(3, 2)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq3_wave2" class="soundwave"/><br />
                <img src="assets/interpolation/6_reference/spectrogram_step02.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(3, 3)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq3_wave3" class="soundwave"/><br />
                <img src="assets/interpolation/6_reference/spectrogram_step03.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(3, 4)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq3_wave4" class="soundwave"/><br />
                <img src="assets/interpolation/6_reference/spectrogram_step04.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(3, 5)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq3_wave5" class="soundwave"/><br />
                <img src="assets/interpolation/6_reference/spectrogram_step05.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(3, 6)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq3_wave6" class="soundwave"/><br />
                <img src="assets/interpolation/6_reference/spectrogram_step06.png"/>
            </td>
        </tr>
    </table>
</div>


### Interpolation example 5

<div class="figure">
    <table>
        <tr>
            <th></th>
            <th>Start preset<br/>"Revers1"</th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th>End preset<br />"SuperGrand"</th>
        </tr>
        <tr>
            <th></th>
            <td>Step 1/7</td>
            <td>Step 2/7</td>
            <td>Step 3/7</td>
            <td>Step 4/7</td>
            <td>Step 5/7</td>
            <td>Step 6/7</td>
            <td>Step 7/7</td>
        </tr>
        <tr> <!-- SPINVAE interp -->
            <th scope="row">
                <button type="button" onclick="onPlaySequenceButtonClicked(8)">
                    <img src="assets/svg/play.svg" class="play_button"/>  <br> Play all
                </button>
                <br>  <br> SPINVAE-2 <br> &nbsp;
            </th>
            <td>
                <button type="button" onclick="onPlayButtonClicked(8, 0)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq8_wave0" class="soundwave"/><br />
                <img src="assets/interpolation/302_spinvae/spectrogram_step00.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(8, 1)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq8_wave1" class="soundwave"/><br />
                <img src="assets/interpolation/302_spinvae/spectrogram_step01.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(8, 2)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq8_wave2" class="soundwave"/><br />
                <img src="assets/interpolation/302_spinvae/spectrogram_step02.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(8, 3)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq8_wave3" class="soundwave"/><br />
                <img src="assets/interpolation/302_spinvae/spectrogram_step03.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(8, 4)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq8_wave4" class="soundwave"/><br />
                <img src="assets/interpolation/302_spinvae/spectrogram_step04.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(8, 5)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq8_wave5" class="soundwave"/><br />
                <img src="assets/interpolation/302_spinvae/spectrogram_step05.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(8, 6)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq8_wave6" class="soundwave"/><br />
                <img src="assets/interpolation/302_spinvae/spectrogram_step06.png"/>
            </td>
        </tr>
        <tr> <!-- REFERENCE interp -->
            <th scope="row">
                <button type="button" onclick="onPlaySequenceButtonClicked(9)">
                    <img src="assets/svg/play.svg" class="play_button"/>  <br> Play all
                </button>
                <br>  <br> Reference <br> (linear)
            </th>
            <td>
                <button type="button" onclick="onPlayButtonClicked(9, 0)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq9_wave0" class="soundwave"/><br />
                <img src="assets/interpolation/302_reference/spectrogram_step00.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(9, 1)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq9_wave1" class="soundwave"/><br />
                <img src="assets/interpolation/302_reference/spectrogram_step01.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(9, 2)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq9_wave2" class="soundwave"/><br />
                <img src="assets/interpolation/302_reference/spectrogram_step02.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(9, 3)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq9_wave3" class="soundwave"/><br />
                <img src="assets/interpolation/302_reference/spectrogram_step03.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(9, 4)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq9_wave4" class="soundwave"/><br />
                <img src="assets/interpolation/302_reference/spectrogram_step04.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(9, 5)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq9_wave5" class="soundwave"/><br />
                <img src="assets/interpolation/302_reference/spectrogram_step05.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(9, 6)"><img src="assets/svg/play.svg" class="play_button"/></button> &nbsp; 
                <img src="assets/svg/soundwave.svg" id="seq9_wave6" class="soundwave"/><br />
                <img src="assets/interpolation/302_reference/spectrogram_step06.png"/>
            </td>
        </tr>
    </table>
</div>



---

# SPINVAE-2 Extrapolations

### Extrapolation example 1

<div class="figure">
    <table>
        <tr>
            <th></th>
            <th colspan="2" class="centered_th">&lt;--- Extrapolation</th>
            <th>Preset</th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th>Preset</th>
            <th colspan="2" class="centered_th">Extrapolation ---&gt;</th>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td><em>BOUM</em></td>
            <td colspan="5" class="centered_th">&lt;---------- Interpolation ----------&gt;</td>
            <td><em>fuzzerro</em></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <th scope="row">
                <button type="button" onclick="onPlaySequenceButtonClicked(10)">
                    <img src="assets/svg/play.svg" class="play_button"/>  <br> Play all
                </button>
                <br>  <br> SPINVAE-2 <br> &nbsp;
            </th>
            <td>
                <button type="button" onclick="onPlayButtonClicked(10, 0)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/> 
                <img src="assets/svg/soundwave.svg" id="seq10_wave0" class="soundwave"/><br />
                <img src="assets/extrapolation/199874_to_016527/spectrogram-2.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(10, 1)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/>
                <img src="assets/svg/soundwave.svg" id="seq10_wave1" class="soundwave"/><br />
                <img src="assets/extrapolation/199874_to_016527/spectrogram-1.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(10, 2)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/> 
                <img src="assets/svg/soundwave.svg" id="seq10_wave2" class="soundwave"/><br />
                <img src="assets/extrapolation/199874_to_016527/spectrogram00.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(10, 3)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/>
                <img src="assets/svg/soundwave.svg" id="seq10_wave3" class="soundwave"/><br />
                <img src="assets/extrapolation/199874_to_016527/spectrogram01.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(10, 4)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/>
                <img src="assets/svg/soundwave.svg" id="seq10_wave4" class="soundwave"/><br />
                <img src="assets/extrapolation/199874_to_016527/spectrogram02.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(10, 5)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/>
                <img src="assets/svg/soundwave.svg" id="seq10_wave5" class="soundwave"/><br />
                <img src="assets/extrapolation/199874_to_016527/spectrogram03.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(10, 6)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/>
                <img src="assets/svg/soundwave.svg" id="seq10_wave6" class="soundwave"/><br />
                <img src="assets/extrapolation/199874_to_016527/spectrogram04.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(10, 7)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/>
                <img src="assets/svg/soundwave.svg" id="seq10_wave7" class="soundwave"/><br />
                <img src="assets/extrapolation/199874_to_016527/spectrogram05.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(10, 8)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/>
                <img src="assets/svg/soundwave.svg" id="seq10_wave8" class="soundwave"/><br />
                <img src="assets/extrapolation/199874_to_016527/spectrogram06.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(10, 9)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/>
                <img src="assets/svg/soundwave.svg" id="seq10_wave9" class="soundwave"/><br />
                <img src="assets/extrapolation/199874_to_016527/spectrogram07.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(10, 10)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/>
                <img src="assets/svg/soundwave.svg" id="seq10_wave10" class="soundwave"/><br />
                <img src="assets/extrapolation/199874_to_016527/spectrogram08.png"/>
            </td>
        </tr>
    </table>
</div>



### Extrapolation example 2

<div class="figure">
    <table>
        <tr>
            <th></th>
            <th colspan="2" class="centered_th">&lt;--- Extrapolation</th>
            <th>Preset</th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th>Preset</th>
            <th colspan="2" class="centered_th">Extrapolation ---&gt;</th>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td><em>INDIA 1</em></td>
            <td colspan="5" class="centered_th">&lt;---------- Interpolation ----------&gt;</td>
            <td><em>Tonewheel2</em></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <th scope="row">
                <button type="button" onclick="onPlaySequenceButtonClicked(11)">
                    <img src="assets/svg/play.svg" class="play_button"/>  <br> Play all
                </button>
                <br>  <br> SPINVAE-2 <br> &nbsp;
            </th>
            <td>
                <button type="button" onclick="onPlayButtonClicked(11, 0)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/> 
                <img src="assets/svg/soundwave.svg" id="seq11_wave0" class="soundwave"/><br />
                <img src="assets/extrapolation/006777_to_246833/spectrogram-2.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(11, 1)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/>
                <img src="assets/svg/soundwave.svg" id="seq11_wave1" class="soundwave"/><br />
                <img src="assets/extrapolation/006777_to_246833/spectrogram-1.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(11, 2)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/> 
                <img src="assets/svg/soundwave.svg" id="seq11_wave2" class="soundwave"/><br />
                <img src="assets/extrapolation/006777_to_246833/spectrogram00.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(11, 3)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/>
                <img src="assets/svg/soundwave.svg" id="seq11_wave3" class="soundwave"/><br />
                <img src="assets/extrapolation/006777_to_246833/spectrogram01.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(11, 4)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/>
                <img src="assets/svg/soundwave.svg" id="seq11_wave4" class="soundwave"/><br />
                <img src="assets/extrapolation/006777_to_246833/spectrogram02.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(11, 5)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/>
                <img src="assets/svg/soundwave.svg" id="seq11_wave5" class="soundwave"/><br />
                <img src="assets/extrapolation/006777_to_246833/spectrogram03.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(11, 6)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/>
                <img src="assets/svg/soundwave.svg" id="seq11_wave6" class="soundwave"/><br />
                <img src="assets/extrapolation/006777_to_246833/spectrogram04.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(11, 7)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/>
                <img src="assets/svg/soundwave.svg" id="seq11_wave7" class="soundwave"/><br />
                <img src="assets/extrapolation/006777_to_246833/spectrogram05.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(11, 8)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/>
                <img src="assets/svg/soundwave.svg" id="seq11_wave8" class="soundwave"/><br />
                <img src="assets/extrapolation/006777_to_246833/spectrogram06.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(11, 9)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/>
                <img src="assets/svg/soundwave.svg" id="seq11_wave9" class="soundwave"/><br />
                <img src="assets/extrapolation/006777_to_246833/spectrogram07.png"/>
            </td>
            <td>
                <button type="button" onclick="onPlayButtonClicked(11, 10)"><img src="assets/svg/play.svg" class="play_button"/></button> <br/>
                <img src="assets/svg/soundwave.svg" id="seq11_wave10" class="soundwave"/><br />
                <img src="assets/extrapolation/006777_to_246833/spectrogram08.png"/>
            </td>
        </tr>
    </table>
</div>

