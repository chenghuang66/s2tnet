# s2tnet

### [Paper](https://proceedings.mlr.press/v157/chen21a.html) 
- This is the official implementation of the paper: **S2TNet: Spatio-Temporal Transformer Networks for Trajectory Prediction in Autonomous Driving** (ACML 2021).

## Quick Start

Requires:

* adamod==0.0.3
* ConfigArgParse==1.5.2
* numpy==1.19.0
* PyYAML==6.0
* scipy==1.7.1
* tensorboardX==2.5.1
* torch==1.9.0
* tqdm==4.31.1

### 1) Install Packages

``` bash
 pip install -r requirements.txt
```

### 2) Dataset

We use [Apollo Scape Trajectory dataset](http://apolloscape.auto/trajectory.html)

## Performance

Results on Apollo Scape:

<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow">WSADE</th>
    <th class="tg-c3ow">ADEv</th>
    <th class="tg-c3ow">ADEp</th>
    <th class="tg-c3ow">ADEb</th>
    <th class="tg-c3ow">WSFDE</th>
    <th class="tg-c3ow">FDEv</th>
    <th class="tg-c3ow">FDEp</th>
    <th class="tg-c3ow">FDEb</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">1.1679</td>
    <td class="tg-c3ow">1.9874</td>
    <td class="tg-c3ow">0.6834</td>
    <td class="tg-c3ow">1.7000</td>
    <td class="tg-c3ow">2.1798</td>
    <td class="tg-c3ow">3.5783</td>
    <td class="tg-c3ow">1.3048</td>
    <td class="tg-c3ow">3.2151</td>
  </tr>
</tbody>
</table>

## S2TNet

### Training & Evaluation
You can train our model by below command:
```
python3 main.py --config ./config/apolloscape/train.yaml
```
### Testing & Uploading to Leaderboard
You can test our model by below command:
```
python3 main.py --config ./config/apolloscape/test.yaml
```
The result file, named as prediction_result.zip, is generated after testing phase.
Then, you can directly upload the file to (http://apolloscape.auto/trajectory.html) to obtain the official results.



## Citation
If you find our work useful for your research, please consider citing the paper:
```
@inproceedings{pmlr-v157-chen21a,
  title = 	 {S2TNet: Spatio-Temporal Transformer Networks for Trajectory Prediction in Autonomous Driving},
  author =       {Chen, Weihuang and Wang, Fangfang and Sun, Hongbin},
  booktitle = 	 {Proceedings of The 13th Asian Conference on Machine Learning},
  pages = 	 {454--469},
  year = 	 {2021},
  volume = 	 {157},
  month = 	 {17--19 Nov}
}

```