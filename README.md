# Screen-Space Normal Distribution Function Caching for Consistent Multi-Resolution Rendering of Large Particle Data

![teaser](https://user-images.githubusercontent.com/30932170/65540916-52820400-deda-11e9-9a8c-76e41b3bf777.png)

> This repository contains a GPU based implementation of the paper: ["Screen-Space Normal Distribution Function Caching for Consistent Multi-Resolution Rendering of Large Particle Data"](https://ieeexplore.ieee.org/abstract/document/8017605) 
> The paper is implemented in C++, OpenGL and GLSL in Visual Studio on a Windows machine.

### Prerequisites
This code was tested with:
- Visual Studio 2015
- OpenGL 4.5
- Windows OS

### Setup

- Clone repository into a local destination on your machine.
- Place any data set you wish to load in the folder 'data', data set must be in '.mmpld' format.  The projects loads the 'laserAblation' data set by default, available in the 'data' folder.

### Running the Solution

- Open the solution 'PointImposters/NdfImposters/NdfImposters.sln'.
- Set the project 'NdfImposterLibraryTests' as the start-up project.
- Build the solution.
- Run the project.

### Navigation and Control

#### General Application
| Functionality | Control |
| ------ | ------ |
| Zoom in/out | hold left mouse button and move  |
| Pan | hold middle mouse button and move |
| Rotate |hold right mouse button and move|
| Light direction | hold 'ctrl'+ left mouse button and move |
| Increase/decrease particle radius | '+'/'-' key |
| Toggle transfer functions | 'm'|
| Toggle tile visibility | 't'|
| Toggle sampling pixel once | '1'|
| Toggle SNDFS binning method | '8'|

#### SNDFS Explorer Widget

| Functionality | Control |
| ------ | ------ |
| Toggle SNFS explorer Widget | 'm' then 'V'|
|Increase/decrease inner circle radius | left click on grey circle then hold left click and move|
|Add a new slize | left click in white space in the widget|
|Move slice|left click on slice, then hold left click and move|
|Increase/decrease slice range|left click on slice, then hold right click and move|

<img width="1212" alt="sndfExplorerWidget" src="https://user-images.githubusercontent.com/30932170/65540429-5eb99180-ded9-11e9-9baf-295afed650a3.png">

### Citation
###
    @article{Ibrahim2018SNDFs,
    title = {Screen-Space Normal Distribution Function Caching for Consistent Multi-Resolution Rendering of Large Particle Data},
    author = {Ibrahim, Mohamed and Wickenh\"{a}user, Patrick and Rautek, Peter and Reina, Guido and Hadwiger, Markus},
    journal = {IEEE Transactions on Visualization and Computer Graphics (Proceedings IEEE Scientific Visualization 2017)},
    year = {2018}
    volume = {24},
    number = {1},
    pages = {944--953}
    }

### Project Page
http://vccvisualization.org/research/sndfs/

### Contact
mohamed.ibrahim@kaust.edu.sa

