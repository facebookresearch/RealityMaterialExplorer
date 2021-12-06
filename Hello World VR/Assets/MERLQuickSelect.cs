using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class MERLQuickSelect : MonoBehaviour
{
	
	public string slot;
	public int index;
	public MasterChangeActiveObject target;
    // Start is called before the first frame update
    void Start()
    {
		index = PlayerPrefs.GetInt(slot+"_matnum",index);
        GetComponentInChildren<Text>().text = target.getMERLname(index);
    }
	public void quickSave(){
		index = target.getMERLnumber();
		GetComponentInChildren<Text>().text = target.getMERLname(index);
		PlayerPrefs.SetInt(slot+"_matnum",index);
	}
	
	public void quickSelect(){
		target.changeMERLMaterial(index);
	}
    // Update is called once per frame
    void Update()
    {
        
    }
}
