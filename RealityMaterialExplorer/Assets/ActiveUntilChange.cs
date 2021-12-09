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
