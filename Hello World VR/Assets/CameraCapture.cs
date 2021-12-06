using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System.IO;

public class CameraCapture : MonoBehaviour
{
    public int fileCounter;
    public KeyCode screenshotKey;
    public Camera _camera;
    
 
    private void LateUpdate()
    {
        if (Input.GetKeyDown(screenshotKey))
        {
            Capture();
        }
    }
 
    public void Capture()
    {
        ScreenCapture.CaptureScreenshot("screenshot");
    }
}
