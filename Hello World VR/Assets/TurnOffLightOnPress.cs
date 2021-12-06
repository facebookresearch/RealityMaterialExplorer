using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TurnOffLightOnPress : MonoBehaviour
{
	Light light;
    // Start is called before the first frame update
    void Start()
    {
        light = GetComponent<Light>();
    }

    // Update is called once per frame
    void Update()
    {
        OVRInput.Update();
		light.intensity = 1.0f-(float)OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger, OVRInput.Controller.LTouch)
		+5.0f*(float)OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, OVRInput.Controller.LTouch);

    }
}
