## Semi-heuristic phase compensation algorithm

The proposed Semi-heuristic phase compensation (SHPC) method is an alternative approach for reconstructing phase maps with reduced processing time compare with the brute-force algorithms while avoiding the local-minimum problem of the heuristic proposals. This algorithm is based on nested searches in which the grid size in every iteration is systematically reduced to optimize the compensation time. This algorithm also have a dynamic version called D-SHPC for compensating holographic videos of dynamic samples. Both algorithms are accurate in phase reconstructions and fast enough to compensate full FOV (1280x960 pixels) holograms at rates of 5 FPS.

In the next image, we present the flowchart of the SHPC

![SHPC Flowchart](https://raw.githubusercontent.com/sobandov/SHPC/gh-pages/SHPC_flowchart.png "SHPC Flowchart")

## Examples
In the following section, you can see some examples of holographic videos and static holograms compensated with thw D-SHPC algorithm. 
The results reported in the following sections were obtained from running the algorithm on a computer supported by an Intel® Xeon® CPU E3-1270 v5 @ 3.60 GHz with 64 GB of RAM.

### Static samples

![SHPC static results](https://raw.githubusercontent.com/sobandov/SHPC/gh-pages/Static_results_SHPC.png "SHPC static results")



### Dynamic samples

#### Smearing Red Blood Cells
<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/oN_x9qtwUy0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</p>  

#### Spatially dense Smearing Red Blood Cells
<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/q1RZo6z9k2w" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</p>  

#### Human sperm
<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/254SkoXl11w" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</p>  

## Code 

### SHPC
Click here for download the SHPC code for MATLAB. 
* [Download MATLAB script](https://drive.google.com/file/d/1lD9ztyp4y3C8RYvCuQzkb_T7Gziqa3NP/view?usp=sharing)

### D-SHPC
Click here for download the D-SHPC code for MATLAB. 
* [Download MATLAB script](copy the link here)


## Funding


## Citation
If using this information for publication, please kindly cite the following paper:

## Support or Contact 

| Researcher  | email | Google Scholar | 
| ------------- | ------------- |-------------| 
| Sofia Obando-Vasquez | *sobandov@eafit.edu.co* |  | 
| Ana Doblas| *adoblas@memphis.edu* | [AnaGoogle](https://scholar.google.es/citations?user=PvvDEMYAAAAJ&hl=en) |
| Carlos Trujillo| *catrujilla@eafit.edu.co* | [CarlosGoogle](https://scholar.google.com/citations?user=BKVrl2gAAAAJ&hl=es) |

This Research is a collaboration between Doblas’ and Trujillos’ research lab.

