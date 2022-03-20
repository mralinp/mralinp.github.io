---
layout: main
title: Home
---

# Most recent posts

<div class="line"></div>

<div class="container">
    <div class="col-md-12 col-lg-12">
{% for post in site.posts limit: 5 %}
<article class="post vt-post">
    <div class="row">
        <div class="col-xs-12 col-sm-5 col-md-5 col-lg-4">
            <div class="post-type post-img">
                <a href="{{post.url}}"><img src="{{ post.img }}" width="200px" height="200px" class="img-responsive" alt="image post"></a>
            </div>
            <div class="author-info author-info-2">
                <ul class="list-inline">
                    <li>
                        <div class="info">
                            <span>
                            <i class="fas fa-calendar"></i> 
                            Posted at: {{ post.date | date: "%-d %B %Y" }}
                            </span>
                            <br>
                            <span>
                            Categories: 
                            {% for cat in post.categories %}
                                {{ cat }},
                            {%endfor%}
                            </span>
                        </div>
                    </li>
                </ul>
            </div>
        </div>
        <div class="col-xs-12 col-sm-7 col-md-7 col-lg-8">
            <div class="caption">
                <h3 class="md-heading"><a href="{{post.url}}">{{ post.title }}</a></h3>
                <p>{{ post.abstract }}</p>
                <a class="btn btn-primary" href="{{post.url}}" role="button">Read More</a> </div>
        </div>
    </div>
</article>
<div class="line"></div>

{% endfor %}   
        
<div class="pagination-wrap">
    <nav aria-label="Page navigation example">
        <ul class="pagination">
        <li class="page-item"><a class="page-link" href="#">Previous</a></li>
        <li class="page-item"><a class="page-link" href="#">1</a></li>
        <li class="page-item"><a class="page-link" href="#">2</a></li>
        <li class="page-item"><a class="page-link" href="#">3</a></li>
        <li class="page-item"><a class="page-link" href="#">Next</a></li>
        </ul>
    </nav>
</div>
<div class="clearfix"></div>
</div>
</div>
