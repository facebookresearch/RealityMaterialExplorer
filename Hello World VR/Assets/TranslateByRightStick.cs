using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TranslateByRightStick : MonoBehaviour
{
	public bool activeMovement;
	private bool dragObject;
	private Vector3 prevPos;
	private Vector3 currentPos;
	private Quaternion prevRotation;
	private Quaternion currentRotation;
	private Vector3 displacement;
	
    // Start is called before the first frame update
	
    void Start()
    {
        dragObject=false;
    }

    // Update is called once per frame
    void Update()
    {	
		//reset dragObject if not dragging any more
		
		
		prevRotation =  currentRotation;
		OVRInput.Update();
		currentPos = OVRInput.GetLocalControllerPosition(OVRInput.Controller.RTouch);
		currentRotation = OVRInput.GetLocalControllerRotation(OVRInput.Controller.RTouch);
		
		if(!activeMovement)return;
		if((OVRInput.Get(OVRInput.Axis1D.SecondaryHandTrigger)>0)){
			if(!dragObject){
				displacement = transform.position - currentPos;
				dragObject= true;
			}
			else{transform.position = currentPos + displacement;
			Quaternion rotate = currentRotation * Quaternion.Inverse(prevRotation);
			transform.Rotate(rotate.eulerAngles, Space.World);}
		}
		else{dragObject=false;}
		return;
		/*
		OVRInput.Update();
		Vector2 direction2 = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick);
		Vector3 direction3 = new Vector3(direction2.x, 0.0f, direction2.y);
		transform.Translate(0.01f * direction3);
		if(OVRInput.Get(OVRInput.RawButton.Y)){transform.Translate(0.005f*Vector3.up);}
		if(OVRInput.Get(OVRInput.RawButton.X)){transform.Translate(0.005f*Vector3.down);}
		if(OVRInput.Get(OVRInput.Button.PrimaryThumbstick)){
			transform.position = new Vector3(0.0f,0.0f,0.83f);
		*/
		
		//Vector3 displacement = currentPos - prevPos;
		//
		
		//transform.Translate(displacement, Space.World);
		//
    }
}
