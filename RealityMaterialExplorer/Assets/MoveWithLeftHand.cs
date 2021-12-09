/*
Copyright (c) Meta Platforms, Inc. and affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveWithLeftHand : MonoBehaviour
{
	private int sticky;
	
	Vector3 offset = new Vector3(0.0f,0.1f,0.1f);
    // Start is called before the first frame update
    void Start()
    {
        sticky = 0;
    }

    // Update is called once per frame
    void Update()
    {	
		OVRInput.Update();
		if(OVRInput.Get(OVRInput.RawButton.X)){sticky=0;}
		if(OVRInput.Get(OVRInput.RawButton.Y)){sticky=1;}
		if(!(sticky==1)){
			Vector3 local_hand_position = OVRInput.GetLocalControllerPosition(OVRInput.Controller.LTouch);
			Quaternion local_rotation = OVRInput.GetLocalControllerRotation(OVRInput.Controller.LTouch);
			Vector3 global_hand_position =  local_hand_position;
			transform.position = global_hand_position + local_rotation*offset;
		}
    }
}
