#!/usr/bin/env python
# -*- coding: utf-8 -*-


from bincrafters import build_template_default
import os

def _is_shared(build):
    return build.options['HDILib:shared'] == True
    
if __name__ == "__main__":

    builder = build_template_default.get_builder() 
    builder.remove_build_if(_is_shared)  
    builder.run()
