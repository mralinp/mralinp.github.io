---
title: Projects
layout: main
---
<h1> Projects and experiences </h1>
<div class="line"></div>
<div class="container bootstrap snippets bootdey">
        {% for post in site.posts %}
        {% if post.categories.first == 'project' %}
        <div class="panel blog-container">
            <div class="panel-body">
            <div class="image-wrapper text-center">
                <a class="image-wrapper image-zoom cboxElement" href="{{post.url}}">
                <img src="{{post.img}}" class="img-thumbnail rounded" width="100%"  alt="Photo of Blog">
                <div class="image-overlay"></div> 
                </a>
            </div>
            <br>
            <a href="{{post.url}}"><h4>{{post.title}}</h4></a>
            <ul class="post-meta list-inline">
                        <li class="list-inline-item">
                            <i class="fa fa-user-circle-o"></i> {{ post.author }}
                        </li>
                        <li class="list-inline-item">
                            <i class="fa fa-calendar-o"></i> {{ post.date | date: "%-d %B %Y" }}
                        </li>
                        <li class="list-inline-item">
                            <i class="fa fa-tags"></i>
                                {{ post.categories.first }}
                                {% for cat in post.categories offset: 1 %}
                                    -> {{ cat }}
                                {% endfor %}
                        </li>
                    </ul>
            <p class="m-top-sm m-bottom-sm">
                {{ post.content | strip_html | truncatewords: 50 }}
            </p>
            <a href="{{post.url}}" class="btn btn-primary"><i class="fa fa-angle-double-right"></i> Continue reading</a>
        </div>
        <div class="line"></div>
    </div>
     {% endif %}
    {% endfor %}
</div>  