# RealityMaterialExplorer
Licensed under the Apache License, Version 2.0
Copyright (c) Meta Platforms, Inc. and affiliates.
# Description
We present RealityMaterialExplorer in VR,  a Quest 2 app that enables real-time viewing and editing of BRDF models in a simple scene. We are motivated to validate the visual impact of physically based material models in VR/AR and inspired by the Open Source Disney BRDF Explorer for desktop that was fundamental to the development of “Disney BRDF” material model that has become the industry standard for PBR materials.

RealityMaterialExplorer VR is designed for users to evaluate and compare their perception of the realistic material models using 100 measured materials from the MERL database.  It can also be used by researchers and 3D content artists to develop intuition and insight about these 4D BRDF models (the essential function for capturing how surfaces reflect light) that can lead to new research and applications.
# Using this Code
This app has been developed in Unity.  After cloning, you should be able to open the "RealityMaterialExplorer" folder as a Unity project.  We used Unity version 2019.4 (LTS).
# Using the App
IMPORTANT: When you load the app, no objects will be visible.  Use the geometry selection buttons (bottom center) to choose a geometry to view.

The menu systems are all controlled by using the blue laser pointer to point with the right hand and then pressing (A) or the right trigger, which is the main button on the right hand.  Sliders can be moved by pressing and holding the button, or just clicking on a point on the slider bar.

The object can be moved with a grab and drag mechanic using the side trigger on the right hand

The left hand controls the point light.  It can be dimmed (or turned off) by pressing the trigger on the side of the left controller.  It can be made much brighter by pressing the left index finger trigger.  It can be dropped (fixed in place) or reattached to the controller with the x and y buttons.

You can have up to three objects at once.  Any controls you use apply only the active object, chosen using the main control panel (bottom center).

