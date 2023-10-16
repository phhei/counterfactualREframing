<?php
    require "password.php";

    if($_POST and array_key_exists("annotator_ID", $_GET)) {
        $conn_db = new mysqli("localhost", "philipp", $db_password, "philipp_umfrage");
        $stat = $conn_db->error . '<br>'. $conn_db->stat() .'<br>';

        $annotator_ID = $conn_db->real_escape_string($_GET["annotator_ID"]);
        $topic_commit = $conn_db->real_escape_string($_POST["topic"]);
        $original_text_commit = $conn_db->real_escape_string($_POST["original_text"]);
        $counterfactual_commit = $conn_db->real_escape_string($_POST["counterfactual"]);

        if ($conn_db->query("SELECT COUNT(*) FROM ReframeCounterfactualAnswer WHERE annotator_ID = ". $annotator_ID ." and topic = '". $topic_commit . "' and original_text = '". $original_text_commit . "' and counterfactual = '". $counterfactual_commit . "';")->fetch_array(MYSQLI_NUM)[0] >= 1) {
           ;
        } else {
            $result = $conn_db -> query(
                "SELECT frame_short FROM ReframeCounterfactualFrames WHERE topic = '". $topic_commit ."';"
            );
            if ($result === FALSE or $result->num_rows == 0) {
                $a_frames = "";
            } else {
                $a_frames = "";
                while ($row = $result->fetch_assoc()) {
                    $a_frames .=  $conn_db->real_escape_string($_POST[$row["frame_short"]]) ."-";
                }
            }
            $a_fluency = $conn_db->real_escape_string($_POST["a_fluency"]);
            $a_meaning = $conn_db->real_escape_string($_POST["a_meaning"]);

            $time = time() - $_POST["timeStart"];
            $query = "INSERT INTO ReframeCounterfactualAnswer VALUES (".$annotator_ID.", '". $topic_commit ."', '". $original_text_commit ."', '". $counterfactual_commit ."', ". $time .",  '". $a_frames ."', ".$a_fluency .", ".$a_meaning .");";
            if($conn_db->query($query) === false) {
                $error_msg = "WARNING: Your last submission failed unfortunately (". $conn_db->error .")";
            } else {
                $error_msg = null;
            }
        }
        $conn_db->close();
    }

    if(array_key_exists("annotator_ID", $_GET) === FALSE) {
        $topic = "n/a";
        $topic_long ="No annotator-ID -- no topic";
        $original_text = "Please insert an annotator-ID in the URL";
        $counterfactual = "Please insert an annotator-ID in the URL";
        $original_frames = "n/a";
        $target_frames ="n/a";
        $frames = [];
    } else {
        $conn_db = new mysqli("localhost", "philipp", $db_password, "philipp_umfrage");
        $stat = $conn_db->error . '<br>'. $conn_db->stat() .'<br>';

        $annotator_ID = intval($conn_db->real_escape_string($_GET["annotator_ID"]));
        
        $samples_done = $conn_db->query(
            "SELECT COUNT(*) FROM  ReframeCounterfactualAnswer WHERE annotator_ID = ". $annotator_ID .";"
            )->fetch_array(MYSQLI_NUM);
        $samples_done = $samples_done === FALSE ? 0 : (is_null($samples_done) ? 1 : $samples_done[0]);
        $samples_total = $conn_db->query(
            "SELECT COUNT(*) FROM ReframeCounterfactualData;"
            )->fetch_array(MYSQLI_NUM);
        $samples_total =  ($samples_total === FALSE or is_null($samples_total)) ? 1 :  max(1, $samples_total[0]);

	    $result = $conn_db -> query(
            "SELECT * FROM ReframeCounterfactualData U WHERE NOT EXISTS(SELECT * FROM ReframeCounterfactualAnswer WHERE ReframeCounterfactualAnswer.topic = U.topic and ReframeCounterfactualAnswer.original_text = U.original_text and ReframeCounterfactualAnswer.counterfactual = U.counterfactual and annotator_ID = ". $annotator_ID .") LIMIT 1;"
        );
        
        if ($result === FALSE or $result->num_rows == 0) {
            $topic = "n/a";
            $topic_long ="SQL-Error or no more samples";
            $original_text = "Either the SQL-Query failed or you have already annotated all samples. Please contact the administrator in case of doubt.";
            $counterfactual = "Either the SQL-Query failed or you have already annotated all samples. Please contact the administrator in case of doubt.";
            $original_frames = "n/a";
            $target_frames ="n/a";
            $frames = [];
        } else {
            $topic = "loading...";
            $topic_long ="loading...";
            $original_text = "loading...";
            $counterfactual = "loading...";
            $original_frames = "loading...";
            $target_frames ="loading...";
            $frames = [];
            while ($row = $result->fetch_assoc()) {
                $topic = $row["topic"];
                $topic_long = $row["topic_long"];
                $original_text = $row["original_text"];
                $counterfactual = $row["counterfactual"];
                $original_frames = $row["original_frames"];
                $target_frames = $row["target_frames"];
                $frames = $conn_db->query(
                    "SELECT * FROM ReframeCounterfactualFrames WHERE topic = '". $topic ."';"
                );
            }
        }
        
        $conn_db->close();
    }

    $counterfactual = is_null($error_msg) ? $counterfactual : $error_msg;
?>

<!DOCTYPE html>
<head>
    <title>You're in annotation mood (<?php echo $_GET["annotator_ID"]; ?>)</title>
</head>
<body style="text-align: center; max-width: 1000px; margin: auto;">
    <h1>[<?php echo $topic; ?>] <?php echo $topic_long; ?></h1>
    <progress id="progress_annotation" value="<?php echo round($samples_done*1000/$samples_total); ?>" max="1000" title="<?php echo $samples_done .' out of '. $samples_total; ?>"> <?php echo $samples_done ." out of ". $samples_total; ?> </progress>
    <div style="width: 98%; background-color: lightgray; border: 1px solid black; border-radius: 20px; padding: 10px; margin-top: 12px; margin-bottom: 20px; display: inline-block;">
        <b><?php echo $counterfactual; ?></b>
    </div>
    <h2>Let's rate ;)</h2>
    <form action="<?php echo $_SERVER["REQUEST_URI"]; ?>" method="POST" autocomplete="off">
        <h3>Mentioned/ emphasised aspects/ frames in the argument above - <?php if (empty($frames)) { echo "Not given"; } else { echo "Select 0-5 frames (no frame selected means the argument does not fall in any of the categories)"; } ?></h3>
            <?php
                $frames_description = [];
                if (!empty($frames)) {
                    while ($row = $frames->fetch_assoc()) {
                        echo '<label title="'. $row["explanation"] .'"><input type="checkbox" name="'. $row["frame_short"] .'" value="'. $row["frame_short"] .'">'. $row["frame_long"] .'</label>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;';
                        array_push($frames_description, '<li><b>'. $row["frame_long"] .'</b>: '. $row["explanation"] .'</li>');
                    }
                }
            ?>
        <h3>Other rating criteria</h3>
            <h4>Fluency</h4>
                How fluent and grammatical is the argument? Is it understandable/ does it make sense?
                <ol start="1">
                    <li><label><input type="radio" value="1" name="a_fluency"> Does not make sense at all (unfinished/ broken text, no meaning extractable)</label></li>
                    <li><label><input type="radio" value="2" name="a_fluency"> With lots of interpretation it is possible to extract some meaning, but observing significant flaws</label></li>
                    <li><label><input type="radio" value="3" name="a_fluency"> Not good English or hard to follow but somewhat valuable</label></li>
                    <li><label><input type="radio" value="4" name="a_fluency"> Only minor flaws, good to follow</label></li>
                    <li><label><input type="radio" value="5" name="a_fluency"> Fluency and grammar is perfect</label></li>
                </ol>
            <h4>Meaning (contribution)</h4>
                Looking at the content of the argument (... not the grammatical correctness or writing style)...
                <ol start="-1">
                    <li><label><input type="radio" value="-1" name="a_meaning"> Argument is misleading/ nonsense or an obvious tautology/ does not contribute anything to the discussion about <u><?php echo $topic_long; ?></u></label></li>
                    <li><label><input type="radio" value="0" name="a_meaning"> Good point made by the argument, but its <b>meaning</b> significantly different from the argument <i><?php echo $original_text; ?></i></label></li>
                    <li><label><input type="radio" value="1" name="a_meaning"> No valuable (misleading) argument, but its <b>meaning</b> similar to the text <i><?php echo $original_text; ?></i></label></li>
                    <li><label><input type="radio" value="10" name="a_meaning"> Contributing argument <b>and also</b> similar to the argument <i><?php echo $original_text; ?></i> regarding the meaning</label></li>
                </ol>
        <input type="hidden" value="<?php echo $topic; ?>" name="topic">
        <input type="hidden" value="<?php echo str_replace('"', '&quot;', $original_text); ?>" name="original_text">
        <input type="hidden" value="<?php echo str_replace('"', '&quot;', $counterfactual); ?>" name="counterfactual">
        <input type="hidden" value="<?php echo time(); ?>" name="timeStart">
        <br>
        <input type="submit" value=">>> Save & next >>>">
    </form>
    <hr>
    <div style="display: inline-block; width: 100%;">
        <h2>Instructions</h2>
        <p>We ask you to rate arguments and which frames/ aspects are mentioned or emphasised by them.</p>

        <h4>Further details</h4>

        The task is to find the aspect of an argumentative sentence or short text unit.
        An aspect in this task is defined as a sub-topic of discourse in the broader topic of the debate about <?php echo $topic_long; ?>. Several aspects are further defined below.

        General statements of facts or opinions which are not further explained (hard to sort) should be not labeled with any aspect.
        It is possible, that more than one aspect is present in a sentence, please mark all applicable aspects.

        Below, the aspect categories are further explained:

        <ul>
            <?php
                if (empty($frames)) {
                    echo '<li>Not given</li>';
                } else {
                    foreach ($frames_description as $frame_des) {
                        echo $frame_des;
                    }
                }
            ?>
        </ul>

        <h3>Positive and negative examples</h3>

        <u>Argument</u>: <i>The anti-uranium movement has used a wide variety of methods to inform and evolve the community and claim: 'Nuclear reactors are vulnerable to terrorist attack'.</i>

        <h4>Positive (you should do it in such a way)</h4>
        <ul>
            <li>selecting "PUBLIC DEBATE" <b>and</b> "ACCIDENTS/SECURITY" as frames, once because of the mention about the 'anti-uranium movement' as a public movement/ protest and once because of contributing with the thought of terrorist attacks</li>
            <li>carefully reading and, hence, notifying a small grammar mistake: 'and claim<i>s</i>', hence only 4 out of 5 fluency</li>
            <li>strong/ informative contribution to the debate (honouring in Meaning (contribution)) although you maybe PRO nuclear energy</li>
            <li>Examples which arguments are (dis)similar in their <b>meaning</b>:</li>
            <ul>
                <li>Meaning-Similar: The anti-uranium movement informs about the risk of terrorist attacks involving nuclear reactors/ People protesting against nuclear energy (hard times for policy), fearing an accident/ There can be a high environmental damage when nuclear reactors are attacked</li>
                <li>NOT Meaning-Similar: The anti-uranium movement has used a wide variety of methods to inform and evolve the community and claim: 'Nuclear reactors are not vulnerable to terrorist attack'/ People has used a wide variety of methods to inform and claim: 'Nuclear reactors are vulnerable to nuclear reactors'/ Nuclear energy is not good because the costs are high to install such a plant.</li>
            </ul>
        </ul>
        <h4>Negative (no, no, no...)</h4>
        <ul>
            <li>lazy voting (always choosing the same/ middle button (0)) or no frames</li>
            <li>not reading the argument carefully and, hence, missing flaws or mentioned aspects</li>
        </ul>
    </div>
</body>