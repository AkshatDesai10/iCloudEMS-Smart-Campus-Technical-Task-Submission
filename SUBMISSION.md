# iCloudEMS Smart Campus — Technical Task Submission

## 1. How would you scale this from one live camera to 500 cameras streaming at once? Where would it break first?

For 500 cameras, I would not process everything on a single machine. I would distribute the camera streams across multiple GPU-enabled servers and process each stream independently, with a queue or streaming service between the cameras and the processing workers. The first bottleneck would most likely be GPU/compute capacity and network bandwidth, especially if all cameras are high resolution and running continuously.

## 2. How would you avoid double-counting or losing track of a person if they briefly leave the camera's view?

I would use the tracking ID along with a short-term history of previously seen IDs. If a person disappears for a few frames, I would keep their track temporarily instead of immediately treating them as exited. For a larger system, I would also use a re-identification approach so the same person can be matched again if they come back into the camera view.

## 3. How would you handle a camera feed that's consistently blurry or poor quality — flag it, skip it, or something else?

I would first detect and flag the camera as having poor quality instead of silently ignoring it. If the feed stays blurry for a certain period, I would mark it for review and avoid trusting attendance or detection results from that camera. The system could continue monitoring the feed and automatically resume normal processing once the quality improves.

