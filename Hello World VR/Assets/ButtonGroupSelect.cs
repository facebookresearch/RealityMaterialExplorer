using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
//using System.Collections;


public class ButtonGroupSelect : MonoBehaviour
{
	public Button[] buttons;
	private ColorBlock activeBlock;
	private ColorBlock inactiveBlock;
	
	public void selectButton(int i){
		foreach(Button b in buttons){
			b.colors = inactiveBlock;
		}
		
		buttons[i].colors = activeBlock;
		
	}
    // Start is called before the first frame update
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
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
