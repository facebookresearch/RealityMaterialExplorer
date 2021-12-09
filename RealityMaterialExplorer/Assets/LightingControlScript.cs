using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class LightingControlScript : MonoBehaviour
{	//private Scene PointScene;
	//private Scene GraceScene;
	//private Scene UffiziScene;
	//private Scene PetersScene;
	//private Scene GroveScene;
    // Start is called before the first frame update
    void Start()
    {
		
		//PointScene = SceneManager.GetSceneByName("PointScene");
		//GraceScene = SceneManager.GetSceneByName("GraceScene");
		//UffiziScene = SceneManager.GetSceneByName("UffiziScene");
		//PetersScene = SceneManager.GetSceneByName("PetersScene");
		//GroveScene = SceneManager.GetSceneByName("GroveScene");
        
    }
	
	public void chooseVoid(){
		SceneManager.LoadScene("PointScene");
		//SceneManager.SetActiveScene(PointScene);
		//SceneManager.SetActiveScene(SceneManager.GetSceneByName("PointScene"));
	}
	
	public void chooseGrace(){
		SceneManager.LoadScene("GraceScene");
		//SceneManager.SetActiveScene(GraceScene);
		//SceneManager.SetActiveScene(s);
	}
	
	public void chooseUffizi(){
		SceneManager.LoadScene("UffiziScene");
		//SceneManager.SetActiveScene(UffiziScene);
	}
	
	public void choosePeters(){
		SceneManager.LoadScene("PetersScene");
		//SceneManager.SetActiveScene(PetersScene);
	}
	
	public void chooseGrove(){
		SceneManager.LoadScene("GroveScene");
		Resources.UnloadUnusedAssets();
		//SceneManager.SetActiveScene(GroveScene);
	}

    // Update is called once per frame
    void Update()
    {
        
    }
}
