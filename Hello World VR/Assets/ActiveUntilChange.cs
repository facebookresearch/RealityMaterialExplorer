using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
//using System.Collections;

public class ActiveUntilChange : MonoBehaviour
{
	private ColorBlock activeBlock;
	private ColorBlock inactiveBlock;
	public Button me;
    // Start is called before the first frame update
	
	public void applyChange(){
		me.colors = inactiveBlock;
	}
	
	public void buttonPressed(){
		me.colors = activeBlock;
	}
	
	
	
    void Start()
    {
        inactiveBlock = new ColorBlock();
		inactiveBlock = ColorBlock.defaultColorBlock;
		inactiveBlock.normalColor = new Color(0.67f,0.67f,0.67f,1.0f);
		inactiveBlock.highlightedColor = new Color(0.75f,0.75f,1.0f,1.0f);
		activeBlock = new ColorBlock();
		activeBlock = ColorBlock.defaultColorBlock;
		activeBlock.normalColor = new Color(1,1,1,1);
		activeBlock.highlightedColor = new Color(0.75f,0.75f,1.0f,1.0f);
		buttonPressed();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
