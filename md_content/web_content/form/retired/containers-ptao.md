+++
date = "2021-04-10T23:59:16-05:00"
tags = ["microservices","containers","docker"]
categories = ["forms"]
images = [""]
author = "Staff"
description = ""
title = "Container Service Request"
draft = true
type = "form"
private = true
+++

<!-- <p id="support-greeting" style="font-style:italic;font-size:120%;" value=""></p> -->
<form action="https://uvarc-api.pods.uvarc.io/rest/general-support-request/" method="post" id="request-form" accept-charset="UTF-8">
<div class="alert" id="response_message" role="alert" style="padding-bottom:0px;">
  <p id="form_post_response"></p>
</div>
<div>
  <input type="hidden" id="category" name="category" value="DCOS">
  <input type="hidden" id="request_title" name="request_title" value="Container Service Request" />
{{% form-userinfo %}}
  <hr size=1 />
  <div class="form-item form-group form-item form-type-select form-group"> <label class="control-label" for="classification">Classification <span class="form-required" title="This field is required.">*</span></label>
    <select required="required" class="form-control form-select required" title="Faculty, postdoctoral associates, and full-time research staff are eligible to request allocations.  " data-toggle="tooltip" id="classification" name="classification"><option value="" selected="selected">- Select -</option><option value="faculty">Faculty</option><option value="staff">Staff</option><option value="postdoc">Postdoctoral Associate</option><option value="other">Other</option></select>
  </div>
  <div class="form-item form-group form-type-select form-group"> 
    <label class="control-label" for="classification">Affiliation <span class="form-required" title="This field is required.">*</span></label>
    <select required="required" class="form-control form-select required" title="Please select the UVA school / department with which you are primarily affiliated." data-toggle="tooltip" id="classification" name="classification">
      <option value="" selected="selected">- Select -</option>
      <option value="cas">College of Arts & Sciences</option>
      <option value="dsi">School of Data Science</option>
      <option value="seas">School of Engineering and Applied Sciences</option>
      <option value="som">School of Medicine</option>
      <option value="darden">Darden School of Business</option>
      <option value="health-system">UVA Health System</option>
      <option value="other">Other</option>
    </select>
  </div>
  <hr size=1 />
  <div class="form-item form-group form-item form-type-textarea form-group"> 
    <label class="control-label" for="project-summary">Project Summary </label>
    <div class="form-textarea-wrapper resizable"><textarea class="form-control form-textarea" id="project-summary" name="project-summary" cols="60" rows="10"></textarea>
    </div>
    <small id="project-summary-Help" class="form-text text-muted">Please describe your project and the container images you want to run.</small>
  </div>
  <hr size=1 />
  <div class="row">
  <div class="col form-item form-group form-item form-type-radios form-group"> 
    <label class="control-label" for="type-of-request">Tier of Service <span class="form-required" title="This field is required.">*</span></label>
    <div id="type-of-request" class="form-radios">
      <div class="form-item form-type-radio radio">
        <input required="required" type="radio" id="tier-1" name="dcos-tier" value="dcos-tier-1" class="form-radio" /> &nbsp; <= 5 containers ($5/month total)</label>
      </div>
      <div class="form-item form-type-radio radio">
        <input required="required" type="radio" id="tier-2" name="dcos-tier" value="dcos-tier-2" class="form-radio" /> &nbsp; 6 - 15 containers ($10/month total)</label>
      </div>
      <div class="form-item form-type-radio radio">
        <input required="required" type="radio" id="tier-3" name="dcos-tier" value="dcos-tier-3" class="form-radio" /> &nbsp; > 15 containers ($48/month total)</label>
      </div>
    </div>
  </div>
  </div>
  <div style="font-size:90%;" class="alert alert-success"><b>Billing Tiers</b> are selected and paid for by the PI. Submit this form again if you wish to change your tier. Stopped containers do not incur charges, nor does local cluster storage or remote NFS mounts to <code>/project</code> storage. Project storage pricing can be found <a href="/userinfo/storage/" style="font-weight:bold;">here</a>.</div>
  <hr size=1 />
  <div class="row">
  <div class="col form-item form-group form-item form-type-radios form-group"> 
    <label class="control-label" for="storage-options">Storage <span class="form-required" title="This field is required.">*</span></label>
    <div id="storage-options" class="form-radios">
      <div class="form-item form-type-radio radio">
        <input required="required" type="radio" id="storage-choice1" name="storage-choice" value="project" class="form-radio" /> &nbsp; No storage required</label>
      </div>
      <div class="form-item form-type-radio radio">
        <input required="required" type="radio" id="storage-choice3" name="storage-choice" value="value" class="form-radio" /> &nbsp; Persistent cluster storage required</label>
      </div>
      <div class="form-item form-type-radio radio">
        <input required="required" type="radio" id="storage-choice4" name="storage-choice" value="zfs" class="form-radio" /> &nbsp; NFS mount of project storage is required</label>
      </div>
    </div>
  </div>
    <div class="col form-item form-group">
      <label class="control-label" for="capacity">Storage Capacity (GB)</label>
      <input class="form-control" type="number" min="0" max="50" id="capacity" name="capacity" value="0" style="width:8rem;" />
      <p class=tiny>The size of storage if required. Specify in 1GB increments.</p>
    </div>
  </div>
  <div class="row">
  <div class="col form-item form-group form-item form-type-radios form-group"> 
    <label class="control-label" for="ssl-required">SSL/HTTPS Required <span class="form-required" title="This field is required.">*</span></label>
    <div id="storage-options" class="form-radios">
      <div class="form-item form-type-radio radio">
        <input checked required="required" type="radio" id="ssl-required-no" name="ssl-required" value="ssl-no" class="form-radio" /> &nbsp; No</label>
      </div>
      <div class="form-item form-type-radio radio">
        <input required="required" type="radio" id="ssl-required-yes" name="ssl-required" value="ssl-yes" class="form-radio" /> &nbsp; Yes</label>
      </div>
    </div>
  </div>
  <div class="col form-item form-group form-item form-type-radios form-group"> 
    <label class="control-label" for="netbadge-required">Netbadge Authentication <span class="form-required" title="This field is required.">*</span></label>
    <div id="storage-options" class="form-radios">
      <div class="form-item form-type-radio radio">
        <input checked required="required" type="radio" id="netbadge-required-no" name="netbadge-required" value="netbadge-no" class="form-radio" /> &nbsp; No</label>
      </div>
      <div class="form-item form-type-radio radio">
        <input required="required" type="radio" id="netbadge-required-yes" name="netbadge-required" value="netbadge-yes" class="form-radio" /> &nbsp; Yes</label>
      </div>
    </div>
  </div>
  </div>
  <hr size=1 />
  <label class="control-label" for="data-sensitivity-2">PTAO <span class="form-required" title="This field is required.">*</span></label>
  <div class="row">
    <div class="col form-item form-type-textarea form-group">
      <input class="form-control form-text required" type="text" id="ptao1" name="ptao1" value="" size="10" maxlength="10" />
    </div>
    <div class="col form-item form-type-textarea form-group">
      <input class="form-control form-text required" type="text" id="ptao2" name="ptao2" value="" size="10" maxlength="10" />
    </div>
    <div class="col form-item form-type-textarea form-group">
      <input class="form-control form-text required" type="text" id="ptao3" name="ptao3" value="" size="10" maxlength="10" />
    </div>
    <div class="col form-item form-type-textarea form-group">
      <input class="form-control form-text required" type="text" id="ptao4" name="ptao4" value="" size="10" maxlength="10" />
    </div>
    <div class="col form-item form-type-textarea form-group">
    </div>
    <div class="col form-item form-type-textarea form-group">
    </div>
  </div>
  <div class="form-item form-group form-type-textarea"> 
    <label class="control-label" for="financial-contact">Financial Contact <span class="form-required" title="This field is required.">*</span></label>
    <input class="form-control form-text required" type="text" id="financial-contact" name="financial-contact" value="" size="200" maxlength="200" />
    <small id="financialContactHelp" class="form-text text-muted">Please enter the name and email address of your financial contact.</small>
  </div>
  <hr size=1 />
  <div class="form-check form-item form-group">
    <label class="control-label" for="data-agreement">Data Agreement <span class="form-required" title="This field is required.">*</span></label>
    <label class="form-check-label" for="data-agreement">
      The owner of these services assumes all responsibility for complying with state, federal, and international data retention laws. Researchers may be required to keep data securely stored for years after a project has ended and should plan accordingly. University of Virginia researchers are strongly encouraged to use the <a href="https://recordsmanagement.virginia.edu/urma/overview" target="_new" style="font-weight:bold;">University Records Management Application (URMA)</a>, a web-based tool that automatically tracks when data can be safely transferred or destroyed.
    </label>
  </div>
  <div class="form-item form-group">
    <input class="form-check-input required" style="margin-left:4rem;" type="checkbox" value="" id="data-agreement">&nbsp;&nbsp; I understand
  </div>
  <div class="form-actions" id="submit-div" style="margin-top:1rem;">
    <hr size="1" style="" />
    <p style="font-size:80%;">Please submit the form only once. If you receive an error message after submitting this request, please check your email to confirm that the submission completed.</p>
    <button class="button-primary btn btn-primary form-submit" id="submit" type="submit" name="op" value="Submit" disabled>Submit</button>
  </div>
</div>
</form>
<div>
</div>

<script type="text/javascript" src="/js/user-session.js"></script>
<script type="text/javascript" src="/js/response-message.js"></script>
