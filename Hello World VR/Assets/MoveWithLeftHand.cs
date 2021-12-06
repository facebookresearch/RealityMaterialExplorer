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
