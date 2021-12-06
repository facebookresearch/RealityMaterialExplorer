using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CopyMaterialEveryFrame : MonoBehaviour
{
	public int myindex;
	public MasterChangeActiveObject master;
	public ChangeActiveObject target;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
		//~GetComponent<Renderer> ().material;
		if(myindex==master.activeIndex){
			GetComponent<Renderer> ().material = new Material(target.activeMaterialForPlot);
			GetComponent<Renderer> ().material.SetFloat("_plot_mode",1.0f);
		}
    }
}
