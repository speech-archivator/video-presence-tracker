# Speech Archivator
**Speech Archivator** deals with problems emerging from [deepfake](https://en.wikipedia.org/wiki/Deepfake) technology. 
Advances in deepfake technology made it possible to manipulate or generate visual and audio content with a high potential to deceive (see [this](https://www.youtube.com/watch?v=cQ54GDm1eL0)).
While this is a fascinating technology, it is also dangerous. 
For example political powers can misuse this technology to create fake videos that are indistinguishable from the real ones. Such videos will cause false beliefs and opinions, just like typical fake news do. 

The ultimate goal of **Speech Archivator** is to watch out for live stream videos including speeches of persons that have a high impact on society. 
We use an artificial neural network to detect faces in the videos.
The video segments containing specific persons are then uploaded to safe decentralized storage such as [IPFS](https://ipfs.io/), where it can't be manipulated and anyone can always retrieve the original video.  

#### Example
![](example.gif)

> This project was created during [HackFS hackathon](https://hackfs.com/) 
> organized by [ETH Global](https://ethglobal.co/).

# video-presence-tracker

A set of Python scripts for cutting out segments from videos containing specified faces.
These segments can be uploaded to the [IPFS](https://ipfs.io/) using the ```ipfs_uploader.py```.
The system uses face detection and face recognition implemented in PyTorch.

## Installation

1. Clone the project:
    ```bash
    git clone https://github.com/speech-archivator/video-presence-tracker.git
    ```

2. Enter the directory and install the dependencies:
    ```bash
    cd video-presence-tracker
    pip install -r requirements.txt
    ```

3. Create a dataset with identities to track.
    The data has to be in the standard format:
    * dataset
        * name1
            * image1.jpg
            * image2.jpg
            * image3.jpg
        * name2
            * image1.jpg
            * image2.jpg
            * ...

    Each image has to contain exactly one face.
    If the face is not detected in any of the images corresponding to the identity, new images will have to be provided.

4. Set the DATASET constant in the ```config.py``` file aiming at the dataset root folder.

5. Finally, execute the ```setup.py``` script:
    ```bash
    python ./setup.py
    ```
   This command downloads the model weights and computes representative feature vectors from the
   provided dataset.

## Usage
- To process a specific video, run the following command:
    ```bash
    python ./process_video.py --display-video --video-path path/to/video.mp4
    ```
    If the ```--display-video``` flag is present, the video will be displayed in a window as it is processed.
    
- The ```track_yt_channel.py``` continually processes new videos uploaded to a YouTube channel specified by the ```CHANNEL_ID```.
    To execute the script run the command bellow:
    ```bash
    python ./track_yt_channel.py --display-video --channel-id CHANNEL_ID --yt-api-key API_KEY
    ```
    To obtain the ```CHANNEL_ID``` go to [this site](https://socialnewsify.com/get-channel-id-by-username-youtube/)
    and enter the channel name.
    The get the YouTube API key read the Data API [documentation](https://developers.google.com/youtube/v3/getting-started). 