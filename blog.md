--- 
layout: main 
title: Blog 
--- 

<section>
    <div class="text-center mb-5">
        <h2 class="display-20 display-md-18 display-lg-16">Most recent blog posts</h2>
    </div>
    <div class="line"></div>
    <div class="row">
        {% for post in site.posts %}
            <div class="col-lg-4 col-md-6 mb-2-6">
                <article class="card card-style2">
                    <div class="card-img">
                    <div class="fill">
                        <a class="image-wrapper image-zoom cboxElement" href="{{post.url}}">
                            <img src="{{post.img}}" class="rounded-top" alt="Photo of Blog">
                        </a>
                    </div>
                        <div class="date"><span>{{ post.date | date: "%b" }}</span>{{ post.date | date: "%Y" }}</div>
                    </div>
                    <div class="card-body">
                        <h3 class="h4"><a href="{{post.url}}">{{post.title}}</a></h3>
                        <p class="display-30">{{ post.content | strip_html | truncatewords: 20 }}</p>
                        <a href="{{post.url}}" class="btn"><i class="fa fa-angle-double-right"></i> Read more</a>
                    </div>
                    <div class="card-footer">
                        <ul>
                        <li><i class="fa fa-user-circle-o"></i> {{ post.author }}</li>                            
                        <li><i class="fa fa-tags"></i>{{ post.categories.first }}</li>
                        </ul>
                    </div>
                </article>
            </div>
        {% endfor %}
    </div>
</section>