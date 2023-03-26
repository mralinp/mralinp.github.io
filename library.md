---
title: Books library
layout: main
---
<section>
    <div class="row">
        {% for post in site.posts %}
            {% if post.categories.first == 'book' %}
                <div class="col-lg-4 col-md-6 mb-2-6">
                    <article class="card card-style2">
                        <div class="card-img">
                        <div class="fill hoverwrap">
                            <a class="image-wrapper image-zoom cboxElement" href="{{post.url}}">
                                <img src="{{post.img}}" class="rounded-top" alt="Photo of Blog">
                            </a>
                            <div class="hovercap">{{ post.title }}</div>
                        </div>
                            <div class="date"><span>{{ post.date | date: "%b" }}</span>{{ post.date | date: "%Y" }}</div>
                        </div>
                        <div class="card-body">
                            <h3 class="h4"><a href="{{post.url}}" title="{{ post.title }}">{{post.title | strip_html | truncate: 35}}</a></h3>
                            <p class="display-30">{{ post.brief | strip_html | truncate: 130 }}</p>
                            <a href="{{post.url}}" class="btn"><i class="fa fa-angle-double-right"></i> Read more</a>
                        </div>
                        <div class="card-footer">
                            <ul>
                            <li><i class="fa fa-user-circle-o"></i> {{ post.author }}</li>                            
                            <li ><i class="fa fa-tags"></i><span class="category">{{ post.categories[1] }}</span></li>
                            </ul>
                        </div>
                    </article>
                </div>
            {% endif %}
        {% endfor %}
    </div>
</section>